from typing import Dict
from torch import Tensor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from .encoder_decoder import EncoderDecoder3D
import copy, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmdet.models.utils import rename_loss_dict
from collections import namedtuple, deque
from mmdet3d.structures import PointData

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, k):
        return random.sample(self.memory, k)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, obs_dim=4, n_actions=3):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

def _all_can_update(flag: bool, device: torch.device) -> bool:
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())
    return flag

@MODELS.register_module()
class MinkUNetWeatherDropper(EncoderDecoder3D):
    def __init__(self, n_observations=None, use_gmx_lite: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cache = {}
        self.learnable_drop = self.train_cfg.get('learnable_drop', False)
        self.use_gmx_lite = bool(use_gmx_lite or self.train_cfg.get('use_gmx_lite', False))
        self.init_dqn_parameters()
        self.n_observations = int(4 if self.use_gmx_lite else 3) if n_observations is None else int(n_observations)
        self.policy_net = DQN(self.n_observations, len(self.drop_bins))
        self.target_net = DQN(self.n_observations, len(self.drop_bins))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False
        if not self.learnable_drop:
            for p in self.policy_net.parameters():
                p.requires_grad = False
        self.memory = ReplayMemory(self.replay_memory_size)
        self.steps_done = 0
        self.dqn_opt = torch.optim.Adam(self.policy_net.parameters(), lr=self.train_cfg.get('dqn_lr', 1e-3), weight_decay=self.train_cfg.get('dqn_weight_decay', 0.0))
        self.dqn_grad_clip = self.train_cfg.get('dqn_grad_clip', 1.0)

    def init_dqn_parameters(self):
        self.BATCH_SIZE = self.train_cfg.get('dqn_batch_size', 32)
        self.GAMMA = self.train_cfg.get('dqn_gamma', 0.99)
        self.EPS_START = self.train_cfg.get('dqn_eps_start', 0.9)
        self.EPS_END = self.train_cfg.get('dqn_eps_end', 0.05)
        self.EPS_DECAY = self.train_cfg.get('dqn_eps_decay', 1000)
        self.TAU = self.train_cfg.get('dqn_tau', 0.005)
        self.replay_memory_size = self.train_cfg.get('dqn_replay_memory_size', 10000)
        self.drop_bins = self.train_cfg.get('dqn_drop_bins', [0.1, 0.2, 0.3])
        self.gmx_k = self.train_cfg.get('gmx_k', 20)
        self.region_voxel_size = self.train_cfg.get('region_voxel_size', 1.0)
        self.gt_fg_label_min = self.train_cfg.get('gt_fg_label_min', 1)
        self.gt_penalty_lambda = self.train_cfg.get('gt_penalty_lambda', 0.5)

    def calculate_reward(self, loss_before, loss_after, gt_ratio):
        return (loss_after - loss_before) - self.gt_penalty_lambda * gt_ratio

    def freeze(self, module):
        if module is None:
            return
        for p in module.parameters(recurse=True):
            p.requires_grad = False

    def unfreeze(self, module):
        if module is None:
            return
        for p in module.parameters(recurse=True):
            p.requires_grad = True

    def gmx_lite_feats(self, points, K, chunk=2048, window=256):
        pos = points[:, :3]
        N = pos.size(0)
        if N == 0 or K <= 0:
            return pos.new_zeros((0, 2))
        az = torch.atan2(pos[:, 1], pos[:, 0])
        order = torch.argsort(az)
        pos_s = pos[order]
        pad = min(window, N)
        pos_c = torch.cat([pos_s[-pad:], pos_s, pos_s[:pad]], 0)
        nei = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            center = pos_s[s:e]
            cand = pos_c[s:s + 2 * pad + (e - s)]
            if cand.size(0) == 0:
                nei.append(torch.zeros((center.size(0), K), dtype=torch.long, device=pos.device))
                continue
            d = torch.cdist(center, cand)
            k_eff = min(K + 1, cand.size(0))
            tk = torch.topk(d, k=k_eff, largest=False).indices
            idx_local = tk[:, 1:K + 1] if tk.size(1) >= K + 1 else tk[:, :K]
            idx_local = (idx_local + s).clamp_(0, N - 1)
            nei.append(order[idx_local])
        nei_idx = torch.cat(nei, 0)
        neigh = pos[nei_idx]
        mu = neigh.mean(dim=1)
        diff = pos - mu
        d1 = diff.norm(p=1, dim=1, keepdim=True)
        d2 = (neigh - mu.unsqueeze(1)).norm(p=2, dim=2).mean(dim=1, keepdim=True)
        s = torch.quantile(d2, 0.9).clamp_min(1e-4)
        d1n = torch.clamp(d1 / (s + 1e-4), 0.0, 5.0)
        d2n = torch.clamp(d2 / (s + 1e-4), 0.0, 5.0)
        return torch.cat([d1n, d2n], dim=1)

    def legacy_point_feats(self, points):
        pos = points[:, :3]
        r = torch.norm(pos[:, :2], dim=1, keepdim=True)
        grid = torch.floor(pos / max(self.region_voxel_size, 1.0)).long()
        key = grid[:, 0] * 73856093 ^ grid[:, 1] * 19349663 ^ grid[:, 2] * 83492791
        uniq, inv = torch.unique(key, return_inverse=True)
        counts = torch.bincount(inv, minlength=uniq.numel()).float()
        dens = counts[inv].unsqueeze(1).log1p()
        r = torch.clamp(r / (r.mean() + 1e-4), 0.0, 5.0)
        dens = torch.clamp(dens / (dens.mean() + 1e-4), 0.0, 5.0)
        return torch.cat([r, dens], dim=1)

    def propose_regions(self, points, pfeat):
        pos, dev = points[:, :3], points.device
        grid = torch.floor(pos / self.region_voxel_size).long()
        key = grid[:, 0] * 73856093 ^ grid[:, 1] * 19349663 ^ grid[:, 2] * 83492791
        uniq, inv = torch.unique(key, return_inverse=True)
        regions, feats = [], []
        if self.use_gmx_lite:
            dt = points.dtype
            gfeat = pfeat
            for rid in range(uniq.size(0)):
                idx = (inv == rid).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                regions.append(idx)
                rg_g = gfeat[idx]
                d1m = rg_g[:, 0].mean().unsqueeze(0)
                d2m = rg_g[:, 1].mean().unsqueeze(0)
                dens = torch.log1p(torch.tensor(idx.numel(), device=dev, dtype=dt)).unsqueeze(0)
                feats.append(torch.cat([d1m, d2m, dens], 0))
            if len(regions) == 0:
                return [torch.arange(points.size(0), device=dev)], points.new_zeros((1, 3))
            return regions, torch.stack(feats, 0)
        else:
            for rid in range(uniq.size(0)):
                idx = (inv == rid).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                regions.append(idx)
                rg_f = pfeat[idx]
                feats.append(rg_f.mean(dim=0, keepdim=False))
            if len(regions) == 0:
                return [torch.arange(points.size(0), device=dev)], points.new_zeros((1, pfeat.size(1)))
            return regions, torch.stack(feats, 0)

    def select_region_action(self, region_feats, uncertainty):
        R = region_feats.size(0)
        obs = torch.cat([region_feats, uncertainty.expand(R, 1)], 1)
        assert obs.size(1) == self.n_observations
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1.0 * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                q = self.policy_net(obs)
                idx = torch.argmax(q)
            r = torch.div(idx, q.size(1), rounding_mode='floor')
            b = idx % q.size(1)
            return int(r), int(b), obs[r]
        else:
            r = random.randrange(R)
            b = random.randrange(len(self.drop_bins))
            return r, b, obs[r]

    @torch.no_grad()
    def get_drop_points(self, cache):
        loss, uncertainty = cache['loss'], cache['uncertainty']
        batch_inputs, batch_data_samples = cache['batch_inputs'], cache['batch_data_samples']
        enum_list, mask_list, points_list, action_list, state_list, gtr_list = [], [], [], [], [], []
        for i in range(len(batch_inputs['points'])):
            points = batch_inputs['points'][i]
            device, dtype, N = points.device, points.dtype, points.size(0)
            if N == 0:
                continue
            enum = torch.full((N,), i, device=device, dtype=torch.long)
            mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask.to(device)
            if self.use_gmx_lite:
                pfeat = self.gmx_lite_feats(points, self.gmx_k)
            else:
                pfeat = self.legacy_point_feats(points)
            regions, region_feats = self.propose_regions(points, pfeat)
            unc = uncertainty.detach().reshape(1,).to(device=device, dtype=dtype)
            rid, bid, obs = self.select_region_action(region_feats.to(device=device, dtype=dtype), unc)
            drop_p = self.drop_bins[bid]
            keep = torch.ones(N, dtype=torch.bool, device=device)
            ridx = regions[rid]
            if ridx.numel() > 0:
                keep[ridx] = torch.rand(ridx.numel(), device=device) > drop_p
            gt_ratio = (mask[ridx] >= self.gt_fg_label_min).float().mean().to(device) if ridx.numel() > 0 else torch.zeros((), device=device)
            enum_list.append(enum[keep]); mask_list.append(mask[keep]); points_list.append(points[keep])
            action_list.append(torch.tensor([bid], device=device, dtype=torch.long))
            state_list.append(obs.reshape(1, -1))
            gtr_list.append(gt_ratio.view(1, 1))
        if len(enum_list) == 0:
            d = batch_inputs['points'][0].device
            return batch_inputs, batch_data_samples, torch.empty(0, device=d, dtype=torch.long), torch.empty(0, self.n_observations), torch.empty(0, 1)
        enum = torch.cat(enum_list, 0); mask = torch.cat(mask_list, 0); points = torch.cat(points_list, 0)
        action = torch.cat(action_list, 0); state = torch.cat(state_list, 0); gtr = torch.cat(gtr_list, 0)
        batch_inputs_out = {'points': []}; batch_data_samples_out = copy.deepcopy(batch_data_samples)
        for i in range(len(batch_inputs['points'])):
            sel = (enum == i)
            if sel.sum() == 0:
                batch_inputs_out['points'].append(batch_inputs['points'][i])
                batch_data_samples_out[i].set_data({'gt_pts_seg': PointData(**{'pts_semantic_mask': batch_data_samples[i].gt_pts_seg.pts_semantic_mask.to(batch_inputs['points'][i].device)})})
            else:
                batch_inputs_out['points'].append(points[sel])
                batch_data_samples_out[i].set_data({'gt_pts_seg': PointData(**{'pts_semantic_mask': mask[sel]})})
        batch_inputs_out['voxels'] = self.data_preprocessor.voxelize(batch_inputs_out['points'], batch_data_samples_out)
        return batch_inputs_out, batch_data_samples_out, action, state, gtr

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        voxel_dict = batch_inputs_dict['voxels'].copy()
        x = self.backbone(voxel_dict['voxels'], voxel_dict['coors'])
        if self.with_neck:
            x = self.neck(x)
        voxel_dict['voxel_feats'] = x
        return voxel_dict

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList) -> Dict[str, Tensor]:
        losses = {}
        voxel_dict = self.extract_feat(batch_inputs_dict)
        loss_decode = self._decode_head_forward_train(voxel_dict, batch_data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(voxel_dict, batch_data_samples))
        if not self.learnable_drop:
            return losses
        voxel_dict2 = self.extract_feat(batch_inputs_dict)
        with torch.no_grad():
            logits_softmax = F.softmax(self.decode_head(voxel_dict2)['logits'], dim=1)
            self.cache = {
                'batch_inputs': batch_inputs_dict,
                'batch_data_samples': batch_data_samples,
                'loss': loss_decode['decode.loss_ce'].detach(),
                'uncertainty': torch.mean(torch.sum(-logits_softmax * torch.log(logits_softmax + 1e-8), dim=1)).detach(),
            }
            batch_inputs_drop, batch_samples_drop, action, state, gtr = self.get_drop_points(self.cache)
        voxel_dict_drop = self.extract_feat(batch_inputs_drop)
        loss_drop = rename_loss_dict('drop_', self._decode_head_forward_train(voxel_dict_drop, batch_samples_drop))
        losses.update(loss_drop)
        with torch.no_grad():
            logits_softmax_drop = F.softmax(self.decode_head(voxel_dict_drop)['logits'], dim=1)
            uncertainty_drop = torch.mean(torch.sum(-logits_softmax_drop * torch.log(logits_softmax_drop + 1e-8), dim=1)).detach()
            next_state = torch.cat([state[:, :self.n_observations - 1], uncertainty_drop.expand(state.size(0), 1)], 1)
            reward = self.calculate_reward(loss_decode['decode.loss_ce'].detach(), loss_drop['drop_decode.loss_ce'].detach(), gtr.squeeze(1)).reshape(-1, 1)
            self.memory.push(state.detach(), action.detach(), next_state.detach(), reward.detach())
        have_any = (len(self.memory) >= 1)
        nonempty = (state.size(0) > 0)
        device = batch_inputs_dict['points'][0].device
        can_update = _all_can_update(have_any and nonempty, device)
        if can_update:
            loss_policy = self.update_policy_net(batch_inputs_dict, use_all_when_small=True)
            losses.update(loss_policy)
        else:
            losses['dqn_dummy'] = sum(p.sum() for p in self.policy_net.parameters()) * 0.0
        losses['dqn_touch'] = sum(p.float().sum() for p in self.policy_net.parameters()) * 0.0
        return losses

    def update_policy_net(self, batch_inputs_dict: dict, use_all_when_small: bool = False):
        M = len(self.memory)
        device = batch_inputs_dict['points'][0].device
        if M == 0:
            return {'dqn_loss': sum(p.sum() for p in self.policy_net.parameters()) * 0.0}
        k = M if (use_all_when_small and M < self.BATCH_SIZE) else self.BATCH_SIZE
        transitions = self.memory.sample(k)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state, 0).to(device)
        next_state_batch = torch.cat(batch.next_state, 0).to(device)
        action_batch = torch.cat(batch.action, 0).view(-1, 1).to(device)
        reward_batch = torch.cat(batch.reward, 0).to(device)
        if state_batch.numel() == 0 or action_batch.numel() == 0:
            return {'dqn_loss': sum(p.sum() for p in self.policy_net.parameters()) * 0.0}
        self.policy_net.train()
        q_s = self.policy_net(state_batch)
        if q_s.numel() == 0:
            return {'dqn_loss': sum(p.sum() for p in self.policy_net.parameters()) * 0.0}
        q_sa = q_s.gather(1, action_batch)
        with torch.no_grad():
            q_next = self.target_net(next_state_batch)
            q_max = q_next.max(1, keepdim=True).values
            target = torch.clamp(reward_batch + self.GAMMA * q_max, -10.0, 10.0)
        loss = F.smooth_l1_loss(q_sa, target)
        if torch.isnan(loss) or torch.isinf(loss):
            return {'dqn_loss': sum(p.sum() for p in self.policy_net.parameters()) * 0.0}
        return {'dqn_loss': loss}

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:
        voxel_dict = self.extract_feat(batch_inputs_dict)
        seg_logits_list = self.decode_head.predict(voxel_dict, batch_data_samples)
        for i in range(len(seg_logits_list)):
            seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)
        return self.postprocess_result(seg_logits_list, batch_data_samples)

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: OptSampleList = None) -> Tensor:
        voxel_dict = self.extract_feat(batch_inputs_dict)
        return self.decode_head.forward(voxel_dict)