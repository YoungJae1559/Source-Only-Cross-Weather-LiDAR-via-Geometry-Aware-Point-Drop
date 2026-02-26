# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from .decode_head import Base3DDecodeHead
from mmdet3d.utils.typing_utils import ConfigType


@MODELS.register_module()
class MinkUNetHead(Base3DDecodeHead):
    def __init__(self, batch_first: bool = True, **kwargs) -> None:
        super(MinkUNetHead, self).__init__(**kwargs)
        self.batch_first = batch_first

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        """Build stabilized MLP-based Segmentation Head."""
        return nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),        # 더 안정적인 정규화
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),             # regularization
            nn.Linear(channels, num_classes)
        )

    def forward(self, voxel_dict: dict) -> dict:
        logits = self.cls_seg(voxel_dict['voxel_feats'])
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        voxel_dict['logits'] = logits
        return voxel_dict

    def loss(self, inputs: dict, batch_data_samples: SampleList,
             train_cfg: ConfigType) -> Dict[str, Tensor]:
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def loss_by_feat(self, voxel_dict, batch_data_samples):
        seg_logits = voxel_dict['logits']
        coors = voxel_dict['coors']

        pred_list = []
        label_list = []
        B = len(batch_data_samples)

        for b in range(B):
            vox_mask = (coors[:, 0] == b) if self.batch_first else (coors[:, -1] == b)
            v_logits = seg_logits[vox_mask]
            p2v = voxel_dict['point2voxel_maps'][b].long().to(v_logits.device)

            if p2v.max().item() >= v_logits.size(0):
                vox_ids = torch.nonzero(vox_mask, as_tuple=False).squeeze(1)
                g2l = torch.full((seg_logits.size(0),), -1, dtype=torch.long, device=v_logits.device)
                g2l[vox_ids] = torch.arange(v_logits.size(0), device=v_logits.device)
                p2v = g2l[p2v]

            point_logits = v_logits[p2v]
            point_labels = batch_data_samples[b].gt_pts_seg.pts_semantic_mask.to(point_logits.device)

            pred_list.append(point_logits)
            label_list.append(point_labels)

        pred = torch.cat(pred_list, dim=0)
        label = torch.cat(label_list, dim=0)

        loss_ce = self.loss_decode(pred, label, ignore_index=self.ignore_index)

        if isinstance(loss_ce, dict):
            return loss_ce
        else:
            return {'loss_ce': loss_ce}

    def predict(self, voxel_dict: dict,
                batch_data_samples: SampleList) -> List[Tensor]:
        voxel_dict = self.forward(voxel_dict)
        seg_pred_list = self.predict_by_feat(voxel_dict, batch_data_samples)
        return seg_pred_list

    def predict_by_feat(self, voxel_dict: dict,
                        batch_data_samples: SampleList) -> List[Tensor]:
        seg_logits = voxel_dict['logits']

        seg_pred_list = []
        coors = voxel_dict['coors']
        for batch_idx in range(len(batch_data_samples)):
            if self.batch_first:
                batch_mask = coors[:, 0] == batch_idx
            else:
                batch_mask = coors[:, -1] == batch_idx
            seg_logits_sample = seg_logits[batch_mask]
            point2voxel_map = voxel_dict['point2voxel_maps'][batch_idx].long()
            point_seg_predicts = seg_logits_sample[point2voxel_map]
            seg_pred_list.append(point_seg_predicts)

        return seg_pred_list