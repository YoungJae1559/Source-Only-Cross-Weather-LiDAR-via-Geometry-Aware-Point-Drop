# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial
from typing import List

import torch
from torch import nn, Tensor
from mmengine.model import BaseModule
from mmengine.registry import MODELS

from mmdet3d.models.layers.minkowski_engine_block import (
    IS_MINKOWSKI_ENGINE_AVAILABLE, MinkowskiBasicBlock, MinkowskiBottleneck,
    MinkowskiConvModule)
from mmdet3d.models.layers.sparse_block import (SparseBasicBlock,
                                                SparseBottleneck,
                                                make_sparse_convmodule,
                                                replace_feature)
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.layers.torchsparse import IS_TORCHSPARSE_AVAILABLE
from mmdet3d.models.layers.torchsparse_block import (TorchSparseBasicBlock,
                                                     TorchSparseBottleneck,
                                                     TorchSparseConvModule)
from mmdet3d.utils import OptMultiConfig

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor

if IS_TORCHSPARSE_AVAILABLE:
    import torchsparse

if IS_MINKOWSKI_ENGINE_AVAILABLE:
    import MinkowskiEngine as ME


# -------------------- Minimal DropPath (timm-free) -------------------- #
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep) * random_tensor


# -------------------- CFBlock (sparse-friendly) -------------------- #
class ConvolutionalAttentionSparse(nn.Module):
    """
    Sparse-friendly external-attention-like block over channels only.
    Operates on (N_active, C) without densifying BEV. Approximates CF attention
    spirit from SCTNet by using learnable key/value banks with grouped double norm.
    """
    def __init__(self, in_channels: int, out_channels: int, inter_channels: int = 64, num_heads: int = 8):
        super().__init__()
        k = max(8, inter_channels)
        self.query = nn.Linear(in_channels, k, bias=False)
        self.key   = nn.Linear(in_channels, k, bias=False)
        self.value = nn.Linear(in_channels, k, bias=False)
        self.proj  = nn.Linear(k, out_channels, bias=False)
        self.norm_q = nn.LayerNorm(k)
        self.norm_k = nn.LayerNorm(k)
        self.norm_v = nn.LayerNorm(k)

    def forward(self, x: Tensor) -> Tensor:  # x: (N_active, C)
        q = self.norm_q(self.query(x))
        k = self.norm_k(self.key(x))
        v = self.norm_v(self.value(x))
        # External-attention style: normalize across the token dim via softmax
        attn = torch.softmax(q, dim=0) * torch.softmax(k, dim=0)
        y = (attn * v)  # elementwise reweighting in the shared inter space
        return self.proj(y)


class MLPSparse(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 2.0, drop_rate: float = 0.):
        super().__init__()
        hidden = max(8, int(dim * hidden_ratio))
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop_rate)
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc2(self.drop1(self.act(self.fc1(x))))
        return self.drop2(x)


class CFBlockSparse(BaseModule):
    """
    CFBlock adapted to sparse features (no BEV densification).
    Matches the user CFBlock API: (in_channels, out_channels, num_heads, drop_rate, drop_path_rate)
    """
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 8, drop_rate: float = 0., drop_path_rate: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn  = ConvolutionalAttentionSparse(in_channels, out_channels, inter_channels=64, num_heads=num_heads)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(out_channels)
        self.mlp   = MLPSparse(out_channels, hidden_ratio=2.0, drop_rate=drop_rate)
        self.proj_in = None
        if in_channels != out_channels:
            self.proj_in = nn.Linear(in_channels, out_channels)

    def forward_matrix(self, f: Tensor) -> Tensor:
        # f: (N_active, C)
        res = f
        if self.proj_in is not None:
            res = self.proj_in(res)
        x = res + self.drop_path(self.attn(self.norm1(f)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    # backend-agnostic sparse tensor wrapper
    def forward(self, x):
        if IS_SPCONV2_AVAILABLE and isinstance(x, SparseConvTensor):
            f = x.features
            fused = self.forward_matrix(f)
            return replace_feature(x, fused)
        if IS_TORCHSPARSE_AVAILABLE and isinstance(x, torchsparse.SparseTensor):
            x.F = self.forward_matrix(x.F)
            return x
        if IS_MINKOWSKI_ENGINE_AVAILABLE and isinstance(x, ME.SparseTensor):
            return x.replace_feature(self.forward_matrix(x.F))
        # Fallback by attribute
        if hasattr(x, 'features') and hasattr(x, 'indices'):
            return replace_feature(x, self.forward_matrix(x.features))
        if hasattr(x, 'F') and hasattr(x, 'C'):
            x.F = self.forward_matrix(x.F); return x
        raise RuntimeError('Unknown sparse tensor type for CFBlockSparse')


# ---------------------------- Backbone ---------------------------- #
@MODELS.register_module()
class MinkUNetBackbone(BaseModule):
    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32,
                 num_stages: int = 4,
                 encoder_channels: List[int] = [32, 64, 128, 256],
                 encoder_blocks: List[int] = [2, 2, 2, 2],
                 decoder_channels: List[int] = [256, 128, 96, 96],
                 decoder_blocks: List[int] = [2, 2, 2, 2],
                 block_type: str = 'basic',
                 sparseconv_backend: str = 'torchsparse',
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        assert num_stages == len(encoder_channels) == len(decoder_channels)
        assert sparseconv_backend in ['torchsparse', 'spconv', 'minkowski']
        self.num_stages = num_stages
        self.sparseconv_backend = sparseconv_backend
        if sparseconv_backend == 'torchsparse':
            input_conv = TorchSparseConvModule
            encoder_conv = TorchSparseConvModule
            decoder_conv = TorchSparseConvModule
            residual_block = TorchSparseBasicBlock if block_type == 'basic' else TorchSparseBottleneck
            residual_branch = None
        elif sparseconv_backend == 'spconv':
            if not IS_SPCONV2_AVAILABLE:
                warnings.warn('Spconv 2.x is not available, turn to use spconv 1.x in mmcv.')
            input_conv = partial(make_sparse_convmodule, conv_type='SubMConv3d')
            encoder_conv = partial(make_sparse_convmodule, conv_type='SparseConv3d')
            decoder_conv = partial(make_sparse_convmodule, conv_type='SparseInverseConv3d')
            residual_block = SparseBasicBlock if block_type == 'basic' else SparseBottleneck
            residual_branch = partial(make_sparse_convmodule, conv_type='SubMConv3d', order=('conv', 'norm'))
        else:
            input_conv = MinkowskiConvModule
            encoder_conv = MinkowskiConvModule
            decoder_conv = partial(MinkowskiConvModule, conv_cfg=dict(type='MinkowskiConvNdTranspose'))
            residual_block = MinkowskiBasicBlock if block_type == 'basic' else MinkowskiBottleneck
            residual_branch = partial(MinkowskiConvModule, act_cfg=None)

        self.conv_input = nn.Sequential(
            input_conv(in_channels, base_channels, kernel_size=3, padding=1, indice_key='subm0'),
            input_conv(base_channels, base_channels, kernel_size=3, padding=1, indice_key='subm0')
        )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip_cf = nn.ModuleList()

        encoder_channels.insert(0, base_channels)
        decoder_channels.insert(0, encoder_channels[-1])

        for i in range(num_stages):
            # Encoder stage i
            encoder_layer = [encoder_conv(encoder_channels[i], encoder_channels[i], kernel_size=2, stride=2, indice_key=f'spconv{i+1}')]
            for j in range(encoder_blocks[i]):
                if j == 0 and encoder_channels[i] != encoder_channels[i + 1]:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i], encoder_channels[i + 1],
                            downsample=residual_branch(encoder_channels[i], encoder_channels[i + 1], kernel_size=1) if residual_branch is not None else None,
                            indice_key=f'subm{i+1}'
                        )
                    )
                else:
                    encoder_layer.append(residual_block(encoder_channels[i + 1], encoder_channels[i + 1], indice_key=f'subm{i+1}'))
            self.encoder.append(nn.Sequential(*encoder_layer))

            # Decoder stage i
            decoder_layer = [decoder_conv(decoder_channels[i], decoder_channels[i + 1], kernel_size=2, stride=2, transposed=True, indice_key=f'spconv{num_stages-i}')]
            for j in range(decoder_blocks[i]):
                if j == 0:
                    decoder_layer.append(
                        residual_block(
                            decoder_channels[i + 1] + encoder_channels[-2 - i],
                            decoder_channels[i + 1],
                            downsample=residual_branch(
                                decoder_channels[i + 1] + encoder_channels[-2 - i],
                                decoder_channels[i + 1], kernel_size=1
                            ) if residual_branch is not None else None,
                            indice_key=f'subm{num_stages-i-1}'
                        )
                    )
                else:
                    decoder_layer.append(residual_block(decoder_channels[i + 1], decoder_channels[i + 1], indice_key=f'subm{num_stages-i-1}'))
            self.decoder.append(nn.ModuleList([decoder_layer[0], nn.Sequential(*decoder_layer[1:])]))

            # Replace SE1by1 with CFBlock (sparse) acting on concat channels
            se_in = decoder_channels[i + 1] + encoder_channels[-2 - i]
            self.skip_cf.append(CFBlockSparse(in_channels=se_in, out_channels=se_in, num_heads=8, drop_rate=0., drop_path_rate=0.))

    def forward(self, voxel_features: Tensor, coors: Tensor) -> Tensor:
        # Build sparse tensor
        if self.sparseconv_backend == 'torchsparse':
            x = torchsparse.SparseTensor(voxel_features, coors)
        elif self.sparseconv_backend == 'spconv':
            spatial_shape = coors.max(0)[0][1:] + 1
            batch_size = int(coors[-1, 0]) + 1
            x = SparseConvTensor(voxel_features, coors, spatial_shape, batch_size)
        else:
            x = ME.SparseTensor(voxel_features, coors)

        # Encoder
        x = self.conv_input(x)
        laterals = [x]
        for enc in self.encoder:
            x = enc(x)
            laterals.append(x)
        laterals = laterals[:-1][::-1]

        # Decoder + CF skip refine (sparse)
        decoder_outs = []
        for i, dec in enumerate(self.decoder):
            x = dec[0](x)
            if self.sparseconv_backend == 'torchsparse':
                x = torchsparse.cat((x, laterals[i]))
            elif self.sparseconv_backend == 'spconv':
                x = replace_feature(x, torch.cat((x.features, laterals[i].features), dim=1))
            else:
                x = ME.cat(x, laterals[i])
            # CF refine in sparse domain (cheap, no BEV densification)
            x = self.skip_cf[i](x)
            x = dec[1](x)
            decoder_outs.append(x)

        if self.sparseconv_backend == 'spconv':
            return decoder_outs[-1].features
        else:
            return decoder_outs[-1].F