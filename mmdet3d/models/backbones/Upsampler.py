import torch
import torch.nn as nn
from mmengine.registry import MODELS


def _dtype_bytes(x: torch.Tensor) -> int:
    if x.dtype in (torch.float16, torch.bfloat16):
        return 2
    if x.dtype in (torch.float32, torch.int32):
        return 4
    if x.dtype in (torch.float64, torch.int64):
        return 8
    return 4


def knn_interpolate(xyz_src: torch.Tensor,
                    feat_src: torch.Tensor,
                    xyz_tgt: torch.Tensor,
                    k: int = 3,
                    eps: float = 1e-8,
                    max_mem_mb: int = 256,
                    force_float32: bool = True) -> torch.Tensor:
    M, N = xyz_tgt.size(0), xyz_src.size(0)
    C = feat_src.size(1)
    if M == 0:
        return xyz_tgt.new_zeros((0, C))
    if N == 0:
        return xyz_tgt.new_zeros((M, C))
    k = max(1, min(int(k), N))
    s_mask = torch.isfinite(xyz_src).all(dim=1) & torch.isfinite(feat_src).all(dim=1)
    xyz_src = xyz_src[s_mask]
    feat_src = torch.nan_to_num(feat_src[s_mask], nan=0.0, posinf=0.0, neginf=0.0)
    N = xyz_src.size(0)
    if N == 0:
        return xyz_tgt.new_zeros((M, C))
    t_mask = torch.isfinite(xyz_tgt).all(dim=1)
    out = feat_src.new_zeros((M, C))
    if not t_mask.any():
        return out
    k = min(k, N)
    q = xyz_tgt[t_mask]
    s = xyz_src
    if force_float32:
        q = q.float()
        s = s.float()
    all_cs = torch.cat([q, s], dim=0)
    mu = all_cs.mean(dim=0)
    sd = all_cs.std(dim=0)
    sd = torch.where(sd.abs() < eps, torch.full_like(sd, 1.0), sd)
    q = (q - mu) / sd
    s = (s - mu) / sd
    max_vals = int(max_mem_mb * 1024 * 1024 // max(_dtype_bytes(q), 1))
    Qc = max(1, min(q.size(0), max_vals // max(s.size(0), 1)))
    Nc = s.size(0) if Qc * s.size(0) <= max_vals else max(1, max_vals // max(Qc, 1))
    res = feat_src.new_empty((q.size(0), C))
    for qs in range(0, q.size(0), Qc):
        qe = min(qs + Qc, q.size(0))
        q_blk = q[qs:qe]
        best_d, best_idx = None, None
        for ss in range(0, s.size(0), Nc):
            se = min(ss + Nc, s.size(0))
            s_blk = s[ss:se]
            D = torch.cdist(q_blk, s_blk, p=2)
            D = torch.nan_to_num(D, nan=float('inf'), posinf=float('inf'), neginf=float('inf'))
            d_blk, i_blk = torch.topk(D, k, dim=1, largest=False)
            i_blk = i_blk + ss
            if best_d is None:
                best_d, best_idx = d_blk, i_blk
            else:
                d_cat = torch.cat([best_d, d_blk], 1)
                i_cat = torch.cat([best_idx, i_blk], 1)
                best_d, pos = torch.topk(d_cat, k, dim=1, largest=False)
                best_idx = torch.gather(i_cat, 1, pos)
        best_idx = best_idx.clamp_(0, N - 1)
        dmin = best_d.min(dim=1, keepdim=True).values
        scale = best_d.mean(dim=1, keepdim=True).clamp_min(eps)
        d0 = best_d - dmin
        w = torch.softmax(-d0 / scale, dim=1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        rowsum = w.sum(dim=1, keepdim=True)
        bad = (rowsum <= 0) | ~torch.isfinite(rowsum)
        if bad.any():
            w[bad] = 1.0 / k
        neigh = feat_src[best_idx]
        neigh = torch.nan_to_num(neigh, nan=0.0, posinf=0.0, neginf=0.0)
        out_blk = (neigh * w.unsqueeze(-1)).sum(1)
        out_blk = torch.nan_to_num(out_blk, nan=0.0, posinf=0.0, neginf=0.0)
        res[qs:qe] = out_blk
    out[t_mask] = res
    return out


@MODELS.register_module()
class FP_Upsampler(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels=None,
                 out_channels=None,
                 k: int = 3,
                 max_knn_mem_mb: int = 256,
                 min_points: int = 10,
                 use_knn_prob: float = 1.0,
                 clamp_value: float = 6.0,
                 k_new_per_point: int = 0,
                 alpha: float = 0.9):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = []
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        layers = []
        c = self.in_channels
        for h in hidden_channels:
            layers += [nn.Linear(c, h), nn.ReLU(inplace=True), nn.LayerNorm(h)]
            c = h
        if c != self.out_channels:
            layers.append(nn.Linear(c, self.out_channels))
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.final_norm = nn.LayerNorm(self.out_channels)
        self.k = int(k)
        self.max_knn_mem_mb = int(max_knn_mem_mb)
        self.input_adapter = None
        self.min_points = int(min_points)
        self.use_knn_prob = float(use_knn_prob)
        self.clamp_value = float(clamp_value)
        self.k_new = int(k_new_per_point)
        self.alpha = float(alpha)

    def _adapt_in(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c == self.in_channels:
            return x
        if (self.input_adapter is None or
            self.input_adapter.in_features != c or
            self.input_adapter.out_features != self.in_channels):
            self.input_adapter = nn.Linear(c, self.in_channels, bias=False).to(x.device, x.dtype)
        return self.input_adapter(x)

    def _gen_voxel_local(self, coords: torch.Tensor, pc_range: torch.Tensor, voxel_size: torch.Tensor) -> torch.Tensor:
        device = coords.device
        pc_min = pc_range[:3].to(device)
        vs = voxel_size.to(device)
        rel = (coords - pc_min) / vs
        idx = torch.floor(rel)
        base = idx * vs + pc_min
        m = (1.0 - self.alpha) * 0.5
        frac = m + (1.0 - 2.0 * m) * torch.rand(coords.size(0), self.k_new, 3, device=device)
        offs = frac * vs.unsqueeze(0).unsqueeze(0)
        new_points = base.unsqueeze(1) + offs
        new_points = new_points.view(-1, 3)
        return new_points

    def forward(self,
                coords: torch.Tensor,
                feats: torch.Tensor,
                num_downsample: int = None,
                gt_coords: torch.Tensor = None,
                pc_range: torch.Tensor = None,
                voxel_size: torch.Tensor = None):
        if gt_coords is None:
            if self.k_new > 0 and pc_range is not None and voxel_size is not None:
                new_pts = self._gen_voxel_local(coords, pc_range, voxel_size)
                tgt = torch.cat([coords, new_pts], dim=0)
            else:
                tgt = coords
        else:
            tgt = gt_coords
        N, M = coords.size(0), tgt.size(0)
        if N == 0 or M == 0:
            return tgt, tgt.new_zeros((M, self.out_channels), dtype=feats.dtype)
        use_knn = not (N < self.min_points or torch.rand(1).item() > self.use_knn_prob)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        if use_knn:
            fi = knn_interpolate(coords, feats, tgt.to(coords.dtype), k=self.k, max_mem_mb=self.max_knn_mem_mb)
        else:
            idx = torch.randint(0, N, (M,), device=coords.device)
            fi = feats[idx]
        fi = self._adapt_in(fi)
        fo = self.mlp(fi)
        fo = self.final_norm(fo + fi)
        fo = torch.tanh(fo) * self.clamp_value
        fo = torch.nan_to_num(fo, nan=0.0, posinf=0.0, neginf=0.0)
        return tgt, fo