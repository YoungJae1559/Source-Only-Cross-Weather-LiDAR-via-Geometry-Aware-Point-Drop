# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from .seg3d_dataset import Seg3DDataset

@DATASETS.register_module()
class SemanticSTFDataset(Seg3DDataset):
    METAINFO = {
        'classes': (
            'car','bicycle','motorcycle','truck','bus','person','bicyclist',
            'motorcyclist','road','parking','sidewalk','other-ground',
            'building','fence','vegetation','trunk','terrain','pole','traffic-sign'
        ),
        'palette': [[100,150,245],[100,230,245],[30,60,150],[80,30,180],[100,80,250],
                    [155,30,30],[255,40,200],[150,30,90],[255,0,255],[255,150,255],
                    [75,0,75],[175,0,75],[255,200,0],[255,120,50],[0,175,0],
                    [135,60,0],[150,240,80],[255,240,150],[255,0,0]],
        'seg_valid_class_ids': tuple(range(19)),
        'seg_all_class_ids': tuple(range(19)),

        'seg_label_mapping': {
            0:19, 1:19, 10:0, 11:1, 13:4, 15:2, 16:4, 18:3, 20:4,
            21:19, 22:19, 23:19, 30:5, 31:6, 32:7, 40:8, 44:9, 48:10,
            49:11, 50:12, 51:13, 52:19, 60:8, 70:14, 71:15, 72:16,
            80:17, 81:18, 99:19, 252:0, 253:6, 254:5, 255:7, 256:4,
            257:4, 258:3, 259:4
        },
        'max_label': 259,
        'ignore_index': 19
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[Union[str, np.ndarray]] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs)

    def get_seg_label_mapping(self, metainfo):
        # pdb.set_trace()
        seg_label_mapping = np.zeros(metainfo['max_label'] + 1, dtype=np.int64)
        for idx in metainfo['seg_label_mapping']:
            seg_label_mapping[idx] = metainfo['seg_label_mapping'][idx]
        return seg_label_mapping