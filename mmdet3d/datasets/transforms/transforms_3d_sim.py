# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose, RandomResize, Resize
from mmdet.datasets.transforms import (PhotoMetricDistortion, RandomCrop,
                                       RandomFlip)
from mmengine import is_list_of, is_tuple_of

from mmdet3d.models.task_modules import VoxelGenerator
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                                LiDARInstance3DBoxes)
from mmdet3d.structures.ops import box_np_ops
from mmdet3d.structures.points import BasePoints
from .data_augment_utils import noise_per_object_v3_

from mmdet3d.structures.points.lidar_points import LiDARPoints

import copy

@TRANSFORMS.register_module()
class RandomDropPointsColor(BaseTransform):
    def __init__(self, drop_ratio: float = 0.2) -> None:
        assert isinstance(drop_ratio, (int, float)) and 0 <= drop_ratio <= 1, \
            f'invalid drop_ratio value {drop_ratio}'
        self.drop_ratio = drop_ratio

    def transform(self, input_dict: dict) -> dict:
        points = input_dict['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims, \
            'Expect points have color attribute'

        if np.random.rand() > 1.0 - self.drop_ratio:
            points.color = points.color * 0.0
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(drop_ratio={self.drop_ratio})'
        return repr_str

@TRANSFORMS.register_module()
class RandomFlip3D(RandomFlip):
    def __init__(self,
                 sync_2d: bool = True,
                 flip_ratio_bev_horizontal: float = 0.0,
                 flip_ratio_bev_vertical: float = 0.0,
                 flip_box3d: bool = True,
                 **kwargs) -> None:
        super(RandomFlip3D, self).__init__(
            prob=flip_ratio_bev_horizontal, direction='horizontal', **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        self.flip_box3d = flip_box3d
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self,
                            input_dict: dict,
                            direction: str = 'horizontal') -> None:
        assert direction in ['horizontal', 'vertical']
        if self.flip_box3d:
            if 'gt_bboxes_3d' in input_dict:
                if 'points' in input_dict:
                    input_dict['points'] = input_dict['gt_bboxes_3d'].flip(
                        direction, points=input_dict['points'])
                else:
                    # vision-only detection
                    input_dict['gt_bboxes_3d'].flip(direction)
            else:
                input_dict['points'].flip(direction)

        if 'centers_2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['img_shape'][1]
            input_dict['centers_2d'][..., 0] = \
                w - input_dict['centers_2d'][..., 0]
            input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]

    def _flip_on_direction(self, results: dict) -> None:
        if 'flip' not in results:
            cur_dir = self._choose_direction()
        else:
            if results['flip']:
                assert 'flip_direction' in results, 'flip and flip_direction '
                'must exist simultaneously'
                cur_dir = results['flip_direction']
            else:
                cur_dir = None
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            self._flip(results)

    def transform(self, input_dict: dict) -> dict:
        if 'img' in input_dict:
            super(RandomFlip3D, self).transform(input_dict)

        if self.sync_2d and 'img' in input_dict:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio_bev_horizontal else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str

@TRANSFORMS.register_module()
class RandomJitterPoints(BaseTransform):
    def __init__(self,
                 jitter_std: List[float] = [0.01, 0.01, 0.01],
                 clip_range: List[float] = [-0.05, 0.05],
                 reflectance_noise=None,
                 exe_prob = 1.0) -> None:
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(jitter_std, seq_types):
            assert isinstance(jitter_std, (int, float)), \
                f'unsupported jitter_std type {type(jitter_std)}'
            jitter_std = [jitter_std, jitter_std, jitter_std]
        self.jitter_std = jitter_std

        if clip_range is not None:
            if not isinstance(clip_range, seq_types):
                assert isinstance(clip_range, (int, float)), \
                    f'unsupported clip_range type {type(clip_range)}'
                clip_range = [-clip_range, clip_range]
        self.clip_range = clip_range
        self.reflectance_noise = reflectance_noise

        self.exe_prob = exe_prob

    def transform(self, input_dict: dict) -> dict:
        if np.random.rand() > self.exe_prob:
            return input_dict

        points = input_dict['points']
        jitter_std = np.array(self.jitter_std, dtype=np.float32)
        jitter_noise = \
            np.random.randn(points.shape[0], 3) * jitter_std[None, :]
        if self.clip_range is not None:
            jitter_noise = np.clip(jitter_noise, self.clip_range[0],
                                   self.clip_range[1])

        points.translate(jitter_noise)
        if self.reflectance_noise is not None:
            reflectance_noise_ = np.random.randn(points.shape[0], 1) * self.reflectance_noise
            points.tensor[:,-1] = torch.tensor(np.clip(points[:,-1].numpy() + reflectance_noise_, 0, 1), dtype=torch.float32).reshape(-1,)
        return input_dict

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(jitter_std={self.jitter_std},'
        repr_str += f' clip_range={self.clip_range})'
        return repr_str

# Range Jittering    
@TRANSFORMS.register_module()
class RandomRangeJitterPoints(BaseTransform):
    def __init__(self,
                 jitter_std: float = 0.01,
                 clip_range: List[float] = [-0.05, 0.05],
                 reflectance_noise=None) -> None:
        seq_types = (list, tuple, np.ndarray)
        self.jitter_std = jitter_std

        if clip_range is not None:
            if not isinstance(clip_range, seq_types):
                assert isinstance(clip_range, (int, float)), \
                    f'unsupported clip_range type {type(clip_range)}'
                clip_range = [-clip_range, clip_range]
        self.clip_range = clip_range
        self.reflectance_noise = reflectance_noise

    def transform(self, input_dict: dict) -> dict:
        points = input_dict['points']
        jitter_std = np.array([self.jitter_std], dtype=np.float32)
        jitter_noise = \
            np.random.randn(points.shape[0], 1) * jitter_std
        jitter_noise = np.repeat(jitter_noise, 3, axis=1)
        if self.clip_range is not None:
            jitter_noise = np.clip(jitter_noise, self.clip_range[0],
                                   self.clip_range[1])

        points.translate(jitter_noise)
        if self.reflectance_noise is not None:
            reflectance_noise_ = np.random.randn(points.shape[0], 1) * self.reflectance_noise
            points.tensor[:,-1] = torch.tensor(np.clip(points[:,-1].numpy() + reflectance_noise_, 0, 1), dtype=torch.float32).reshape(-1,)
        return input_dict

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(jitter_std={self.jitter_std},'
        repr_str += f' clip_range={self.clip_range})'
        return repr_str

@TRANSFORMS.register_module()
class ObjectSample(BaseTransform):
    def __init__(self,
                 db_sampler: dict,
                 sample_2d: bool = False,
                 use_ground_plane: bool = False) -> None:
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = TRANSFORMS.build(db_sampler)
        self.use_ground_plane = use_ground_plane
        self.disabled = False

    @staticmethod
    def remove_points_in_boxes(points: BasePoints,
                               boxes: np.ndarray) -> np.ndarray:
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def transform(self, input_dict: dict) -> dict:
        if self.disabled:
            return input_dict

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        if self.use_ground_plane:
            ground_plane = input_dict.get('plane', None)
            assert ground_plane is not None, '`use_ground_plane` is True ' \
                                             'but find plane is None'
        else:
            ground_plane = None
        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.numpy(),
                gt_labels_3d,
                img=None,
                ground_plane=ground_plane)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.int64)
        input_dict['points'] = points

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(db_sampler={self.db_sampler},'
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' use_ground_plane={self.use_ground_plane})'
        return repr_str

@TRANSFORMS.register_module()
class ObjectNoise(BaseTransform):
    def __init__(self,
                 translation_std: List[float] = [0.25, 0.25, 0.25],
                 global_rot_range: List[float] = [0.0, 0.0],
                 rot_range: List[float] = [-0.15707963267, 0.15707963267],
                 num_try: int = 100) -> None:
        self.translation_std = translation_std
        self.global_rot_range = global_rot_range
        self.rot_range = rot_range
        self.num_try = num_try

    def transform(self, input_dict: dict) -> dict:
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']

        # TODO: this is inplace operation
        numpy_box = gt_bboxes_3d.numpy()
        numpy_points = points.numpy()

        noise_per_object_v3_(
            numpy_box,
            numpy_points,
            rotation_perturb=self.rot_range,
            center_noise_std=self.translation_std,
            global_random_rot_range=self.global_rot_range,
            num_try=self.num_try)

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.new_box(numpy_box)
        input_dict['points'] = points.new_point(numpy_points)
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_try={self.num_try},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' global_rot_range={self.global_rot_range},'
        repr_str += f' rot_range={self.rot_range})'
        return repr_str

@TRANSFORMS.register_module()
class GlobalAlignment(BaseTransform):
    def __init__(self, rotation_axis: int) -> None:
        self.rotation_axis = rotation_axis

    def _trans_points(self, results: dict, trans_factor: np.ndarray) -> None:
        results['points'].translate(trans_factor)

    def _rot_points(self, results: dict, rot_mat: np.ndarray) -> None:
        results['points'].rotate(rot_mat.T)

    def _check_rot_mat(self, rot_mat: np.ndarray) -> None:
        is_valid = np.allclose(np.linalg.det(rot_mat), 1.0)
        valid_array = np.zeros(3)
        valid_array[self.rotation_axis] = 1.0
        is_valid &= (rot_mat[self.rotation_axis, :] == valid_array).all()
        is_valid &= (rot_mat[:, self.rotation_axis] == valid_array).all()
        assert is_valid, f'invalid rotation matrix {rot_mat}'

    def transform(self, results: dict) -> dict:
        assert 'axis_align_matrix' in results, \
            'axis_align_matrix is not provided in GlobalAlignment'

        axis_align_matrix = results['axis_align_matrix']
        assert axis_align_matrix.shape == (4, 4), \
            f'invalid shape {axis_align_matrix.shape} for axis_align_matrix'
        rot_mat = axis_align_matrix[:3, :3]
        trans_vec = axis_align_matrix[:3, -1]

        self._check_rot_mat(rot_mat)
        self._rot_points(results, rot_mat)
        self._trans_points(results, trans_vec)

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rotation_axis={self.rotation_axis})'
        return repr_str

@TRANSFORMS.register_module()
class GlobalRotScaleTrans(BaseTransform):
    def __init__(self,
                 rot_range: List[float] = [-0.78539816, 0.78539816],
                 scale_ratio_range: List[float] = [0.95, 1.05],
                 translation_std: List[int] = [0, 0, 0],
                 shift_height: bool = False,
                 aug=False) -> None:
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'

        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

        self.aug = aug

    def _trans_bbox_points(self, input_dict: dict) -> None:
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        if 'gt_bboxes_3d' in input_dict:
            input_dict['gt_bboxes_3d'].translate(trans_factor)
        if self.aug == True:
            input_dict['points_aug'].translate(trans_factor)

    def _rot_bbox_points(self, input_dict: dict) -> None:
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            # rotate points with bboxes
            points, rot_mat_T = input_dict['gt_bboxes_3d'].rotate(
                noise_rotation, input_dict['points'])
            input_dict['points'] = points
            if self.aug == True:
                input_dict['points_aug'].rotate(noise_rotation)
        else:
            # if no bbox in input_dict, only rotate points
            rot_mat_T = input_dict['points'].rotate(noise_rotation)
            if self.aug == True:
                input_dict['points_aug'].rotate(noise_rotation)

        input_dict['pcd_rotation'] = rot_mat_T
        input_dict['pcd_rotation_angle'] = noise_rotation

    def _scale_bbox_points(self, input_dict: dict) -> None:
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points

        if self.aug == True:
            points_aug = input_dict['points_aug']
            points_aug.scale(scale)
            if self.shift_height:
                assert 'height' in points_aug.attribute_dims.keys(), \
                    'setting shift_height=True but points_aug have no height attribute'
                points_aug.tensor[:, points_aug.attribute_dims['height']] *= scale
            input_dict['points_aug'] = points_aug

        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            input_dict['gt_bboxes_3d'].scale(scale)

    def _random_scale(self, input_dict: dict) -> None:
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def transform(self, input_dict: dict) -> dict:
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str

@TRANSFORMS.register_module()
class PointShuffle(BaseTransform):

    def transform(self, input_dict: dict) -> dict:
        idx = input_dict['points'].shuffle()
        idx = idx.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[idx]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[idx]

        return input_dict

    def __repr__(self) -> str:
        return self.__class__.__name__

@TRANSFORMS.register_module()
class ObjectRangeFilter(BaseTransform):
    def __init__(self, point_cloud_range: List[float]) -> None:
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: dict) -> dict:
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(bool)]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str

@TRANSFORMS.register_module()
class PointsRangeFilter(BaseTransform):
    def __init__(self, point_cloud_range: List[float]) -> None:
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: dict) -> dict:
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        return input_dict

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str

@TRANSFORMS.register_module()
class ObjectNameFilter(BaseTransform):
    def __init__(self, classes: List[str]) -> None:
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def transform(self, input_dict: dict) -> dict:
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=bool)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


@TRANSFORMS.register_module()
class PointSample(BaseTransform):
    def __init__(self,
                 num_points: int,
                 sample_range: Optional[float] = None,
                 replace: bool = False,
                 save_choices: bool = False) -> None:
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace
        self.save_choices = save_choices

    def _points_random_sampling(
        self,
        points: BasePoints,
        num_samples: Union[int, float],
        sample_range: Optional[float] = None,
        replace: bool = False,
        return_choices: bool = False
    ) -> Union[Tuple[BasePoints, np.ndarray], BasePoints]:
        if isinstance(num_samples, float):
            assert num_samples < 1
            num_samples = int(
                np.random.uniform(self.num_points, 1.) * points.shape[0])

        if not replace:
            replace = (points.shape[0] < num_samples)
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            dist = np.linalg.norm(points.coord.numpy(), axis=1)
            far_inds = np.where(dist >= sample_range)[0]
            near_inds = np.where(dist < sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(
                    far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def transform(self, input_dict: dict) -> dict:
        points = input_dict['points']
        points, choices = self._points_random_sampling(
            points,
            self.num_points,
            self.sample_range,
            self.replace,
            return_choices=True)
        input_dict['points'] = points

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            input_dict['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        if self.save_choices:
            input_dict['subsample_idx'] = choices

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points},'
        repr_str += f' sample_range={self.sample_range},'
        repr_str += f' replace={self.replace})'

        return repr_str

@TRANSFORMS.register_module()
class IndoorPointSample(PointSample):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            'IndoorPointSample is deprecated in favor of PointSample')
        super(IndoorPointSample, self).__init__(*args, **kwargs)


@TRANSFORMS.register_module()
class IndoorPatchPointSample(BaseTransform):
    def __init__(self,
                 num_points: int,
                 block_size: float = 1.5,
                 sample_rate: Optional[float] = None,
                 ignore_index: Optional[int] = None,
                 use_normalized_coord: bool = False,
                 num_try: int = 10,
                 enlarge_size: float = 0.2,
                 min_unique_num: Optional[int] = None,
                 eps: float = 1e-2) -> None:
        self.num_points = num_points
        self.block_size = block_size
        self.ignore_index = ignore_index
        self.use_normalized_coord = use_normalized_coord
        self.num_try = num_try
        self.enlarge_size = enlarge_size if enlarge_size is not None else 0.0
        self.min_unique_num = min_unique_num
        self.eps = eps

        if sample_rate is not None:
            warnings.warn(
                "'sample_rate' has been deprecated and will be removed in "
                'the future. Please remove them from your code.')

    def _input_generation(self, coords: np.ndarray, patch_center: np.ndarray,
                          coord_max: np.ndarray, attributes: np.ndarray,
                          attribute_dims: dict,
                          point_type: type) -> BasePoints:
        centered_coords = coords.copy()
        centered_coords[:, 0] -= patch_center[0]
        centered_coords[:, 1] -= patch_center[1]

        if self.use_normalized_coord:
            normalized_coord = coords / coord_max
            attributes = np.concatenate([attributes, normalized_coord], axis=1)
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(normalized_coord=[
                    attributes.shape[1], attributes.shape[1] +
                    1, attributes.shape[1] + 2
                ]))

        points = np.concatenate([centered_coords, attributes], axis=1)
        points = point_type(
            points, points_dim=points.shape[1], attribute_dims=attribute_dims)

        return points

    def _patch_points_sampling(
            self, points: BasePoints,
            sem_mask: np.ndarray) -> Tuple[BasePoints, np.ndarray]:
        coords = points.coord.numpy()
        attributes = points.tensor[:, 3:].numpy()
        attribute_dims = points.attribute_dims
        point_type = type(points)

        coord_max = np.amax(coords, axis=0)
        coord_min = np.amin(coords, axis=0)

        for _ in range(self.num_try):
            cur_center = coords[np.random.choice(coords.shape[0])]
            cur_max = cur_center + np.array(
                [self.block_size / 2.0, self.block_size / 2.0, 0.0])
            cur_min = cur_center - np.array(
                [self.block_size / 2.0, self.block_size / 2.0, 0.0])
            cur_max[2] = coord_max[2]
            cur_min[2] = coord_min[2]
            cur_choice = np.sum(
                (coords >= (cur_min - self.enlarge_size)) *
                (coords <= (cur_max + self.enlarge_size)),
                axis=1) == 3

            if not cur_choice.any():  # no points in this patch
                continue

            cur_coords = coords[cur_choice, :]
            cur_sem_mask = sem_mask[cur_choice]
            point_idxs = np.where(cur_choice)[0]
            mask = np.sum(
                (cur_coords >= (cur_min - self.eps)) * (cur_coords <=
                                                        (cur_max + self.eps)),
                axis=1) == 3

            if self.min_unique_num is None:
                vidx = np.ceil(
                    (cur_coords[mask, :] - cur_min) / (cur_max - cur_min) *
                    np.array([31.0, 31.0, 62.0]))
                vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 +
                                 vidx[:, 2])
                flag1 = len(vidx) / 31.0 / 31.0 / 62.0 >= 0.02
            else:
                # if `min_unique_num` is provided, directly compare with it
                flag1 = mask.sum() >= self.min_unique_num

            # 2. selected patch should contain enough annotated points
            if self.ignore_index is None:
                flag2 = True
            else:
                flag2 = np.sum(cur_sem_mask != self.ignore_index) / \
                               len(cur_sem_mask) >= 0.7

            if flag1 and flag2:
                break

        # sample idx to `self.num_points`
        if point_idxs.size >= self.num_points:
            # no duplicate in sub-sampling
            choices = np.random.choice(
                point_idxs, self.num_points, replace=False)
        else:
            # do not use random choice here to avoid some points not counted
            dup = np.random.choice(point_idxs.size,
                                   self.num_points - point_idxs.size)
            idx_dup = np.concatenate(
                [np.arange(point_idxs.size),
                 np.array(dup)], 0)
            choices = point_idxs[idx_dup]

        # construct model input
        points = self._input_generation(coords[choices], cur_center, coord_max,
                                        attributes[choices], attribute_dims,
                                        point_type)

        return points, choices

    def transform(self, input_dict: dict) -> dict:
        points = input_dict['points']

        assert 'pts_semantic_mask' in input_dict.keys(), \
            'semantic mask should be provided in training and evaluation'
        pts_semantic_mask = input_dict['pts_semantic_mask']

        points, choices = self._patch_points_sampling(points,
                                                      pts_semantic_mask)

        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask[choices]

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_semantic_mask'] = \
                pts_semantic_mask[choices]

        pts_instance_mask = input_dict.get('pts_instance_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[choices]
            # 'eval_ann_info' will be passed to evaluator
            if 'eval_ann_info' in input_dict:
                input_dict['eval_ann_info']['pts_instance_mask'] = \
                    pts_instance_mask[choices]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points},'
        repr_str += f' block_size={self.block_size},'
        repr_str += f' ignore_index={self.ignore_index},'
        repr_str += f' use_normalized_coord={self.use_normalized_coord},'
        repr_str += f' num_try={self.num_try},'
        repr_str += f' enlarge_size={self.enlarge_size},'
        repr_str += f' min_unique_num={self.min_unique_num},'
        repr_str += f' eps={self.eps})'
        return repr_str

@TRANSFORMS.register_module()
class BackgroundPointsFilter(BaseTransform):
    def __init__(self, bbox_enlarge_range: Union[Tuple[float], float]) -> None:
        assert (is_tuple_of(bbox_enlarge_range, float)
                and len(bbox_enlarge_range) == 3) \
            or isinstance(bbox_enlarge_range, float), \
            f'Invalid arguments bbox_enlarge_range {bbox_enlarge_range}'

        if isinstance(bbox_enlarge_range, float):
            bbox_enlarge_range = [bbox_enlarge_range] * 3
        self.bbox_enlarge_range = np.array(
            bbox_enlarge_range, dtype=np.float32)[np.newaxis, :]

    def transform(self, input_dict: dict) -> dict:
        points = input_dict['points']
        gt_bboxes_3d = input_dict['gt_bboxes_3d']

        # avoid groundtruth being modified
        gt_bboxes_3d_np = gt_bboxes_3d.tensor.clone().numpy()
        gt_bboxes_3d_np[:, :3] = gt_bboxes_3d.gravity_center.clone().numpy()

        enlarged_gt_bboxes_3d = gt_bboxes_3d_np.copy()
        enlarged_gt_bboxes_3d[:, 3:6] += self.bbox_enlarge_range
        points_numpy = points.tensor.clone().numpy()
        foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, gt_bboxes_3d_np, origin=(0.5, 0.5, 0.5))
        enlarge_foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, enlarged_gt_bboxes_3d, origin=(0.5, 0.5, 0.5))
        foreground_masks = foreground_masks.max(1)
        enlarge_foreground_masks = enlarge_foreground_masks.max(1)
        valid_masks = ~np.logical_and(~foreground_masks,
                                      enlarge_foreground_masks)

        input_dict['points'] = points[valid_masks]
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[valid_masks]

        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[valid_masks]
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(bbox_enlarge_range={self.bbox_enlarge_range.tolist()})'
        return repr_str

@TRANSFORMS.register_module()
class VoxelBasedPointSampler(BaseTransform):
    def __init__(self,
                 cur_sweep_cfg: dict,
                 prev_sweep_cfg: Optional[dict] = None,
                 time_dim: int = 3) -> None:
        self.cur_voxel_generator = VoxelGenerator(**cur_sweep_cfg)
        self.cur_voxel_num = self.cur_voxel_generator._max_voxels
        self.time_dim = time_dim
        if prev_sweep_cfg is not None:
            assert prev_sweep_cfg['max_num_points'] == \
                cur_sweep_cfg['max_num_points']
            self.prev_voxel_generator = VoxelGenerator(**prev_sweep_cfg)
            self.prev_voxel_num = self.prev_voxel_generator._max_voxels
        else:
            self.prev_voxel_generator = None
            self.prev_voxel_num = 0

    def _sample_points(self, points: np.ndarray, sampler: VoxelGenerator,
                       point_dim: int) -> np.ndarray:
        voxels, coors, num_points_per_voxel = sampler.generate(points)
        if voxels.shape[0] < sampler._max_voxels:
            padding_points = np.zeros([
                sampler._max_voxels - voxels.shape[0], sampler._max_num_points,
                point_dim
            ],
                                      dtype=points.dtype)
            padding_points[:] = voxels[0]
            sample_points = np.concatenate([voxels, padding_points], axis=0)
        else:
            sample_points = voxels

        return sample_points

    def transform(self, results: dict) -> dict:
        points = results['points']
        original_dim = points.shape[1]
        map_fields2dim = []
        start_dim = original_dim
        points_numpy = points.numpy()
        extra_channel = [points_numpy]
        for idx, key in enumerate(results['pts_mask_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        start_dim += len(results['pts_mask_fields'])
        for idx, key in enumerate(results['pts_seg_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        points_numpy = np.concatenate(extra_channel, axis=-1)
        cur_points_flag = (points_numpy[:, self.time_dim] == 0)
        cur_sweep_points = points_numpy[cur_points_flag]
        prev_sweeps_points = points_numpy[~cur_points_flag]
        if prev_sweeps_points.shape[0] == 0:
            prev_sweeps_points = cur_sweep_points

        # Shuffle points before sampling
        np.random.shuffle(cur_sweep_points)
        np.random.shuffle(prev_sweeps_points)

        cur_sweep_points = self._sample_points(cur_sweep_points,
                                               self.cur_voxel_generator,
                                               points_numpy.shape[1])
        if self.prev_voxel_generator is not None:
            prev_sweeps_points = self._sample_points(prev_sweeps_points,
                                                     self.prev_voxel_generator,
                                                     points_numpy.shape[1])

            points_numpy = np.concatenate(
                [cur_sweep_points, prev_sweeps_points], 0)
        else:
            points_numpy = cur_sweep_points

        if self.cur_voxel_generator._max_num_points == 1:
            points_numpy = points_numpy.squeeze(1)
        results['points'] = points.new_point(points_numpy[..., :original_dim])

        # Restore the corresponding seg and mask fields
        for key, dim_index in map_fields2dim:
            results[key] = points_numpy[..., dim_index]

        return results

    def __repr__(self) -> str:
        def _auto_indent(repr_str, indent):
            repr_str = repr_str.split('\n')
            repr_str = [' ' * indent + t + '\n' for t in repr_str]
            repr_str = ''.join(repr_str)[:-1]
            return repr_str

        repr_str = self.__class__.__name__
        indent = 4
        repr_str += '(\n'
        repr_str += ' ' * indent + f'num_cur_sweep={self.cur_voxel_num},\n'
        repr_str += ' ' * indent + f'num_prev_sweep={self.prev_voxel_num},\n'
        repr_str += ' ' * indent + f'time_dim={self.time_dim},\n'
        repr_str += ' ' * indent + 'cur_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.cur_voxel_generator), 8)},\n'
        repr_str += ' ' * indent + 'prev_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.prev_voxel_generator), 8)})'
        return repr_str


@TRANSFORMS.register_module()
class AffineResize(BaseTransform):
    def __init__(self,
                 img_scale: Tuple,
                 down_ratio: int,
                 bbox_clip_border: bool = True) -> None:

        self.img_scale = img_scale
        self.down_ratio = down_ratio
        self.bbox_clip_border = bbox_clip_border

    def transform(self, results: dict) -> dict:
        if 'center' not in results:
            img = results['img']
            height, width = img.shape[:2]
            center = np.array([width / 2, height / 2], dtype=np.float32)
            size = np.array([width, height], dtype=np.float32)
            results['affine_aug'] = False
        else:
            img = results['img']
            center = results['center']
            size = results['size']

        trans_affine = self._get_transform_matrix(center, size, self.img_scale)

        img = cv2.warpAffine(img, trans_affine[:2, :], self.img_scale)

        if isinstance(self.down_ratio, tuple):
            trans_mat = [
                self._get_transform_matrix(
                    center, size,
                    (self.img_scale[0] // ratio, self.img_scale[1] // ratio))
                for ratio in self.down_ratio
            ]  # (3, 3)
        else:
            trans_mat = self._get_transform_matrix(
                center, size, (self.img_scale[0] // self.down_ratio,
                               self.img_scale[1] // self.down_ratio))

        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['trans_mat'] = trans_mat

        if 'gt_bboxes' in results:
            self._affine_bboxes(results, trans_affine)

        if 'centers_2d' in results:
            centers2d = self._affine_transform(results['centers_2d'],
                                               trans_affine)
            valid_index = (centers2d[:, 0] >
                           0) & (centers2d[:, 0] <
                                 self.img_scale[0]) & (centers2d[:, 1] > 0) & (
                                     centers2d[:, 1] < self.img_scale[1])
            results['centers_2d'] = centers2d[valid_index]

            if 'gt_bboxes' in results:
                results['gt_bboxes'] = results['gt_bboxes'][valid_index]
                if 'gt_bboxes_labels' in results:
                    results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                        valid_index]
                if 'gt_masks' in results:
                    raise NotImplementedError(
                        'AffineResize only supports bbox.')

            if 'gt_bboxes_3d' in results:
                results['gt_bboxes_3d'].tensor = results[
                    'gt_bboxes_3d'].tensor[valid_index]
                if 'gt_labels_3d' in results:
                    results['gt_labels_3d'] = results['gt_labels_3d'][
                        valid_index]

            results['depths'] = results['depths'][valid_index]

        return results

    def _affine_bboxes(self, results: dict, matrix: np.ndarray) -> None:
        bboxes = results['gt_bboxes']
        bboxes[:, :2] = self._affine_transform(bboxes[:, :2], matrix)
        bboxes[:, 2:] = self._affine_transform(bboxes[:, 2:], matrix)
        if self.bbox_clip_border:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0,
                                                       self.img_scale[0] - 1)
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0,
                                                       self.img_scale[1] - 1)
        results['gt_bboxes'] = bboxes

    def _affine_transform(self, points: np.ndarray,
                          matrix: np.ndarray) -> np.ndarray:
        num_points = points.shape[0]
        hom_points_2d = np.concatenate((points, np.ones((num_points, 1))),
                                       axis=1)
        hom_points_2d = hom_points_2d.T
        affined_points = np.matmul(matrix, hom_points_2d).T
        return affined_points[:, :2]

    def _get_transform_matrix(self, center: Tuple, scale: Tuple,
                              output_scale: Tuple[float]) -> np.ndarray:
        src_w = scale[0]
        dst_w = output_scale[0]
        dst_h = output_scale[1]

        src_dir = np.array([0, src_w * -0.5])
        dst_dir = np.array([0, dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2, :] = self._get_ref_point(src[0, :], src[1, :])
        dst[2, :] = self._get_ref_point(dst[0, :], dst[1, :])

        get_matrix = cv2.getAffineTransform(src, dst)

        matrix = np.concatenate((get_matrix, [[0., 0., 1.]]))

        return matrix.astype(np.float32)

    def _get_ref_point(self, ref_point1: np.ndarray,
                       ref_point2: np.ndarray) -> np.ndarray:
        d = ref_point1 - ref_point2
        ref_point3 = ref_point2 + np.array([-d[1], d[0]])
        return ref_point3

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'down_ratio={self.down_ratio}) '
        return repr_str

@TRANSFORMS.register_module()
class RandomShiftScale(BaseTransform):
    def __init__(self, shift_scale: Tuple[float], aug_prob: float) -> None:

        self.shift_scale = shift_scale
        self.aug_prob = aug_prob

    def transform(self, results: dict) -> dict:
        img = results['img']

        height, width = img.shape[:2]

        center = np.array([width / 2, height / 2], dtype=np.float32)
        size = np.array([width, height], dtype=np.float32)

        if random.random() < self.aug_prob:
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)
            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)
            results['affine_aug'] = True
        else:
            results['affine_aug'] = False

        results['center'] = center
        results['size'] = size

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(shift_scale={self.shift_scale}, '
        repr_str += f'aug_prob={self.aug_prob}) '
        return repr_str

@TRANSFORMS.register_module()
class Resize3D(Resize):

    def _resize_3d(self, results: dict) -> None:
        if 'centers_2d' in results:
            results['centers_2d'] *= results['scale_factor'][:2]
        results['cam2img'][0] *= np.array(results['scale_factor'][0])
        results['cam2img'][1] *= np.array(results['scale_factor'][1])

    def transform(self, results: dict) -> dict:
        super(Resize3D, self).transform(results)
        self._resize_3d(results)
        return results


@TRANSFORMS.register_module()
class RandomResize3D(RandomResize):
    def _resize_3d(self, results: dict) -> None:
        if 'centers_2d' in results:
            results['centers_2d'] *= results['scale_factor'][:2]
        results['cam2img'][0] *= np.array(results['scale_factor'][0])
        results['cam2img'][1] *= np.array(results['scale_factor'][1])

    def transform(self, results: dict) -> dict:
        if 'scale' not in results:
            results['scale'] = self._random_scale()
        self.resize.scale = results['scale']
        results = self.resize(results)
        self._resize_3d(results)

        return results

@TRANSFORMS.register_module()
class RandomCrop3D(RandomCrop):
    def __init__(
        self,
        crop_size: tuple,
        crop_type: str = 'absolute',
        allow_negative_crop: bool = False,
        recompute_bbox: bool = False,
        bbox_clip_border: bool = True,
        rel_offset_h: tuple = (0., 1.),
        rel_offset_w: tuple = (0., 1.)
    ) -> None:
        super().__init__(
            crop_size=crop_size,
            crop_type=crop_type,
            allow_negative_crop=allow_negative_crop,
            recompute_bbox=recompute_bbox,
            bbox_clip_border=bbox_clip_border)
        self.rel_offset_h = rel_offset_h
        self.rel_offset_w = rel_offset_w

    def _crop_data(self,
                   results: dict,
                   crop_size: tuple,
                   allow_negative_crop: bool = False) -> dict:
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if 'img_crop_offset' not in results:
                margin_h = max(img.shape[0] - crop_size[0], 0)
                margin_w = max(img.shape[1] - crop_size[1], 0)
                # TOCHECK: a little different from LIGA implementation
                offset_h = np.random.randint(
                    self.rel_offset_h[0] * margin_h,
                    self.rel_offset_h[1] * margin_h + 1)
                offset_w = np.random.randint(
                    self.rel_offset_w[0] * margin_w,
                    self.rel_offset_w[1] * margin_w + 1)
            else:
                offset_w, offset_h = results['img_crop_offset']

            crop_h = min(crop_size[0], img.shape[0])
            crop_w = min(crop_size[1], img.shape[1])
            crop_y1, crop_y2 = offset_h, offset_h + crop_h
            crop_x1, crop_x2 = offset_w, offset_w + crop_w

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # manipulate camera intrinsic matrix
        # needs to apply offset to K instead of P2 (on KITTI)
        if isinstance(results['cam2img'], list):
            # TODO ignore this, but should handle it in the future
            pass
        else:
            K = results['cam2img'][:3, :3].copy()
            inv_K = np.linalg.inv(K)
            T = np.matmul(inv_K, results['cam2img'][:3])
            K[0, 2] -= crop_x1
            K[1, 2] -= crop_y1
            offset_cam2img = np.matmul(K, T)
            results['cam2img'][:offset_cam2img.shape[0], :offset_cam2img.
                               shape[1]] = offset_cam2img

        results['img_crop_offset'] = [offset_w, offset_h]

        return results

    def transform(self, results: dict) -> dict:
        image_size = results['img'].shape[:2]
        if 'crop_size' not in results:
            crop_size = self._get_crop_size(image_size)
            results['crop_size'] = crop_size
        else:
            crop_size = results['crop_size']
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self) -> dict:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border}), '
        repr_str += f'rel_offset_h={self.rel_offset_h}), '
        repr_str += f'rel_offset_w={self.rel_offset_w})'
        return repr_str

@TRANSFORMS.register_module()
class PhotoMetricDistortion3D(PhotoMetricDistortion):
    def transform(self, results: dict) -> dict:
        assert 'img' in results, '`img` is not found in results'
        img = results['img']
        img = img.astype(np.float32)
        if 'photometric_param' not in results:
            photometric_param = self._random_flags()
            results['photometric_param'] = photometric_param
        else:
            photometric_param = results['photometric_param']

        (mode, brightness_flag, contrast_flag, saturation_flag, hue_flag,
         swap_flag, delta_value, alpha_value, saturation_value, hue_value,
         swap_value) = photometric_param

        # random brightness
        if brightness_flag:
            img += delta_value

        if mode == 1:
            if contrast_flag:
                img *= alpha_value

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if saturation_flag:
            img[..., 1] *= saturation_value

        # random hue
        if hue_flag:
            img[..., 0] += hue_value
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if contrast_flag:
                img *= alpha_value

        # randomly swap channels
        if swap_flag:
            img = img[..., swap_value]

        results['img'] = img
        return results


@TRANSFORMS.register_module()
class MultiViewWrapper(BaseTransform):
    def __init__(
        self,
        transforms: dict,
        override_aug_config: bool = True,
        process_fields: list = ['img', 'cam2img', 'lidar2cam'],
        collected_keys: list = [
            'scale', 'scale_factor', 'crop', 'img_crop_offset', 'ori_shape',
            'pad_shape', 'img_shape', 'pad_fixed_size', 'pad_size_divisor',
            'flip', 'flip_direction', 'rotate'
        ],
        randomness_keys: list = [
            'scale', 'scale_factor', 'crop_size', 'img_crop_offset', 'flip',
            'flip_direction', 'photometric_param'
        ]
    ) -> None:
        self.transforms = Compose(transforms)
        self.override_aug_config = override_aug_config
        self.collected_keys = collected_keys
        self.process_fields = process_fields
        self.randomness_keys = randomness_keys

    def transform(self, input_dict: dict) -> dict:
        for key in self.collected_keys:
            if key not in input_dict or \
                    not isinstance(input_dict[key], list):
                input_dict[key] = []
        prev_process_dict = {}
        for img_id in range(len(input_dict['img'])):
            process_dict = {}

            if img_id != 0 and self.override_aug_config:
                for key in self.randomness_keys:
                    if key in prev_process_dict:
                        process_dict[key] = prev_process_dict[key]

            for key in self.process_fields:
                if key in input_dict:
                    process_dict[key] = input_dict[key][img_id]
            process_dict = self.transforms(process_dict)
            # store the randomness variable in transformation.
            prev_process_dict = process_dict

            # store the related results to results_dict
            for key in self.process_fields:
                if key in process_dict:
                    input_dict[key][img_id] = process_dict[key]
            # update the keys
            for key in self.collected_keys:
                if key in process_dict:
                    if len(input_dict[key]) == img_id + 1:
                        input_dict[key][img_id] = process_dict[key]
                    else:
                        input_dict[key].append(process_dict[key])

        for key in self.collected_keys:
            if len(input_dict[key]) == 0:
                input_dict.pop(key)
        return input_dict

@TRANSFORMS.register_module()
class PolarMix(BaseTransform):
    def __init__(self,
                 instance_classes: List[int],
                 swap_ratio: float = 0.5,
                 rotate_paste_ratio: float = 1.0,
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        assert is_list_of(instance_classes, int), \
            'instance_classes should be a list of int'
        self.instance_classes = instance_classes
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def polar_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']

        # 1. swap point cloud
        if np.random.random() < self.swap_ratio:
            start_angle = (np.random.random() - 1) * np.pi  # -pi~0
            end_angle = start_angle + np.pi
            # calculate horizontal angle for each point
            yaw = -torch.atan2(points.coord[:, 1], points.coord[:, 0])
            mix_yaw = -torch.atan2(mix_points.coord[:, 1], mix_points.coord[:, 0])

            # select points in sector
            idx = (yaw <= start_angle) | (yaw >= end_angle)
            mix_idx = (mix_yaw > start_angle) & (mix_yaw < end_angle)

            # swap
            points = points.cat([points[idx], mix_points[mix_idx]])
            pts_semantic_mask = np.concatenate(
                (pts_semantic_mask[idx.numpy()],
                 mix_pts_semantic_mask[mix_idx.numpy()]),
                axis=0)

        # 2. rotate-pasting
        if np.random.random() < self.rotate_paste_ratio:
            # extract instance points
            instance_points, instance_pts_semantic_mask = [], []
            for instance_class in self.instance_classes:
                mix_idx = mix_pts_semantic_mask == instance_class
                instance_points.append(mix_points[mix_idx])
                instance_pts_semantic_mask.append(
                    mix_pts_semantic_mask[mix_idx])
            instance_points = mix_points.cat(instance_points)
            instance_pts_semantic_mask = np.concatenate(
                instance_pts_semantic_mask, axis=0)

            # rotate-copy
            copy_points = [instance_points]
            copy_pts_semantic_mask = [instance_pts_semantic_mask]
            angle_list = [
                np.random.random() * np.pi * 2 / 3,
                (np.random.random() + 1) * np.pi * 2 / 3
            ]
            for angle in angle_list:
                new_points = instance_points.clone()
                new_points.rotate(angle)
                copy_points.append(new_points)
                copy_pts_semantic_mask.append(instance_pts_semantic_mask)
            copy_points = instance_points.cat(copy_points)
            copy_pts_semantic_mask = np.concatenate(
                copy_pts_semantic_mask, axis=0)

            points = points.cat([points, copy_points])
            pts_semantic_mask = np.concatenate(
                (pts_semantic_mask, copy_pts_semantic_mask), axis=0)

        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through PolarMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before polarmix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.polar_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'swap_ratio={self.swap_ratio}, '
        repr_str += f'rotate_paste_ratio={self.rotate_paste_ratio}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str

@TRANSFORMS.register_module()
class LaserMix(BaseTransform):
    def __init__(self,
                 num_areas: List[int],
                 pitch_angles: Sequence[float],
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        assert is_list_of(num_areas, int), \
            'num_areas should be a list of int.'
        self.num_areas = num_areas

        assert len(pitch_angles) == 2, \
            'The length of pitch_angles should be 2, ' \
            f'but got {len(pitch_angles)}.'
        assert pitch_angles[1] > pitch_angles[0], \
            'pitch_angles[1] should be larger than pitch_angles[0].'
        self.pitch_angles = pitch_angles

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def laser_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']

        # convert angle to radian
        pitch_angle_down = self.pitch_angles[0] / 180 * np.pi
        pitch_angle_up = self.pitch_angles[1] / 180 * np.pi

        rho = torch.sqrt(points.coord[:, 0]**2 + points.coord[:, 1]**2)
        pitch = torch.atan2(points.coord[:, 2], rho)
        pitch = torch.clamp(pitch, pitch_angle_down + 1e-5,
                            pitch_angle_up - 1e-5)

        mix_rho = torch.sqrt(mix_points.coord[:, 0]**2 +
                             mix_points.coord[:, 1]**2)
        mix_pitch = torch.atan2(mix_points.coord[:, 2], mix_rho)
        mix_pitch = torch.clamp(mix_pitch, pitch_angle_down + 1e-5,
                                pitch_angle_up - 1e-5)

        num_areas = np.random.choice(self.num_areas, size=1)[0]
        angle_list = np.linspace(pitch_angle_up, pitch_angle_down,
                                 num_areas + 1)
        out_points = []
        out_pts_semantic_mask = []
        for i in range(num_areas):
            start_angle = angle_list[i + 1]
            end_angle = angle_list[i]
            if i % 2 == 0:  # pick from original point cloud
                idx = (pitch > start_angle) & (pitch <= end_angle)
                out_points.append(points[idx])
                out_pts_semantic_mask.append(pts_semantic_mask[idx.numpy()])
            else:  # pickle from mixed point cloud
                idx = (mix_pitch > start_angle) & (mix_pitch <= end_angle)
                out_points.append(mix_points[idx])
                out_pts_semantic_mask.append(
                    mix_pts_semantic_mask[idx.numpy()])
        out_points = points.cat(out_points)
        out_pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)
        input_dict['points'] = out_points
        input_dict['pts_semantic_mask'] = out_pts_semantic_mask
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through LaserMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before lasermix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.laser_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_areas={self.num_areas}, '
        repr_str += f'pitch_angles={self.pitch_angles}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str

GROUND_LABELS = {8, 9, 10, 16}

def _ground_mask(points, z_thr=0.2):
    z = points.coord[:, 2]
    return (z.abs() < z_thr)

@TRANSFORMS.register_module()
class DepthSelectiveJitter(BaseTransform):
    def __init__(self,
                 num_areas: List[int],
                 pitch_angles: Sequence[float],
                 pre_transform: Optional[Sequence[dict]] = None,
                 pre_transform4orig: Optional[Sequence[dict]] = None,
                 pre_transform4orig_prob: float = 1.0,
                 prob: float = 1.0,
                 separately_output=False) -> None:
        assert all(isinstance(x, int) for x in num_areas)
        self.num_areas = num_areas
        assert len(pitch_angles) == 2 and pitch_angles[1] > pitch_angles[0]
        self.pitch_angles = pitch_angles
        self.separately_output = separately_output
        self.prob = prob
        self.pre_transform = Compose(pre_transform) if pre_transform is not None else None
        self.pre_transform4orig_prob = pre_transform4orig_prob
        self.pre_transform4orig = Compose(pre_transform4orig) if pre_transform4orig is not None else None

    def depthselective_transform(self, input_dict: dict, mix_results: dict) -> dict:
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']
        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        pitch_angle_down = self.pitch_angles[0] / 180 * np.pi
        pitch_angle_up = self.pitch_angles[1] / 180 * np.pi
        rho = torch.sqrt(points.coord[:, 0]**2 + points.coord[:, 1]**2)
        pitch = torch.atan2(points.coord[:, 2], rho)
        pitch = torch.clamp(pitch, pitch_angle_down + 1e-5, pitch_angle_up - 1e-5)
        mix_rho = torch.sqrt(mix_points.coord[:, 0]**2 + mix_points.coord[:, 1]**2)
        mix_pitch = torch.atan2(mix_points.coord[:, 2], mix_rho)
        mix_pitch = torch.clamp(mix_pitch, pitch_angle_down + 1e-5, pitch_angle_up - 1e-5)
        num_areas = np.random.choice(self.num_areas, size=1)[0]
        angle_list = np.linspace(pitch_angle_up, pitch_angle_down, num_areas + 1)
        out_points = []
        out_pts_semantic_mask = []
        for i in range(num_areas):
            start_angle = angle_list[i + 1]
            end_angle = angle_list[i]
            b = (start_angle > -2.0*np.pi/180) & (end_angle < 2.0*np.pi/180)
            if i % 2 == 0 or b:
                idx = (pitch > start_angle) & (pitch <= end_angle)
                out_points.append(points[idx])
                out_pts_semantic_mask.append(pts_semantic_mask[idx.numpy()])
            else:
                idx = (mix_pitch > start_angle) & (mix_pitch <= end_angle)
                out_points.append(mix_points[idx])
                out_pts_semantic_mask.append(mix_pts_semantic_mask[idx.numpy()])
        out_points = points.cat(out_points)
        out_pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)
        if self.separately_output:
            input_dict['points_aug'] = out_points
            input_dict['pts_semantic_mask_aug'] = out_pts_semantic_mask
        else:
            input_dict['points'] = out_points
            input_dict['pts_semantic_mask'] = out_pts_semantic_mask
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        if np.random.rand() > self.prob:
            return input_dict
        assert 'dataset' in input_dict
        dataset = input_dict['dataset']
        index = np.random.randint(0, len(dataset))
        mix_input_dict = dataset.get_data_info(index)
        if self.pre_transform4orig is not None and self.pre_transform4orig_prob > np.random.rand():
            input_dict = self.pre_transform4orig(input_dict)
        if self.pre_transform is not None:
            mix_input_dict.update({'dataset': dataset})
            mix_results = self.pre_transform(mix_input_dict)
            mix_results.pop('dataset')
        input_dict = self.depthselective_transform(input_dict, mix_results)
        return input_dict

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(num_areas={self.num_areas}, "
                f"pitch_angles={self.pitch_angles}, pre_transform={self.pre_transform}, prob={self.prob})")

@TRANSFORMS.register_module()
class AngleSelectiveJitter(BaseTransform):
    def __init__(self,
                 instance_classes: List[int],
                 swap_ratio: float = 0.5,
                 rotate_paste_ratio: float = 1.0,
                 pre_transform: Optional[Sequence[dict]] = None,
                 pre_transform4orig: Optional[Sequence[dict]] = None,
                 pre_transform4orig_prob: float = 1.0,
                 prob: float = 1.0,
                 separately_output: bool = False,
                 instance_copy: bool = False,
                 azimuth_multiplier=None) -> None:
        assert all(isinstance(x, int) for x in instance_classes)
        self.instance_classes = instance_classes
        self.instance_copy = instance_copy
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio
        self.separately_output = separately_output
        self.azimuth_range = np.pi if azimuth_multiplier is None else np.pi * azimuth_multiplier
        self.prob = prob
        self.pre_transform = Compose(pre_transform) if pre_transform is not None else None
        self.pre_transform4orig_prob = pre_transform4orig_prob
        self.pre_transform4orig = Compose(pre_transform4orig) if pre_transform4orig is not None else None

    def angleselective_transform(self, input_dict: dict, mix_results: dict) -> dict:
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']
        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        if np.random.random() < self.swap_ratio:
            start_angle = (np.random.random() - 1) * np.pi
            end_angle = start_angle + self.azimuth_range
            yaw = -torch.atan2(points.coord[:, 1], points.coord[:, 0])
            mix_yaw = -torch.atan2(mix_points.coord[:, 1], mix_points.coord[:, 0])
            idx = (yaw <= start_angle) | (yaw >= end_angle)
            mix_idx = (mix_yaw > start_angle) & (mix_yaw < end_angle)
            m = torch.from_numpy(pts_semantic_mask)
            rr = (m[idx.numpy()] == 8).float().mean() if idx.any() else torch.tensor(0.)
            if rr <= 0.4:
                points = points.cat([points[idx], mix_points[mix_idx]])
                pts_semantic_mask = np.concatenate((pts_semantic_mask[idx.numpy()], mix_pts_semantic_mask[mix_idx.numpy()]), axis=0)
        if self.instance_copy and np.random.random() < self.rotate_paste_ratio:
            instance_points, instance_pts_semantic_mask = [], []
            for c in self.instance_classes:
                if c in GROUND_LABELS:
                    continue
                mix_idx = mix_pts_semantic_mask == c
                instance_points.append(mix_points[mix_idx])
                instance_pts_semantic_mask.append(mix_pts_semantic_mask[mix_idx])
            if len(instance_points) > 0:
                instance_points = mix_points.cat(instance_points)
                instance_pts_semantic_mask = np.concatenate(instance_pts_semantic_mask, axis=0)
                copy_points = [instance_points]
                copy_pts_semantic_mask = [instance_pts_semantic_mask]
                angle_list = [np.random.random() * 2*np.pi/3, (np.random.random() + 1) * 2*np.pi/3]
                for angle in angle_list:
                    new_points = instance_points.clone(); new_points.rotate(angle)
                    copy_points.append(new_points)
                    copy_pts_semantic_mask.append(instance_pts_semantic_mask)
                copy_points = instance_points.cat(copy_points)
                copy_pts_semantic_mask = np.concatenate(copy_pts_semantic_mask, axis=0)
                points = points.cat([points, copy_points])
                pts_semantic_mask = np.concatenate((pts_semantic_mask, copy_pts_semantic_mask), axis=0)
        if self.separately_output:
            input_dict['points_aug'] = points
            input_dict['pts_semantic_mask_aug'] = pts_semantic_mask
        else:
            input_dict['points'] = points
            input_dict['pts_semantic_mask'] = pts_semantic_mask
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        if np.random.rand() > self.prob:
            return input_dict
        assert 'dataset' in input_dict
        dataset = input_dict['dataset']
        index = input_dict['sample_idx']
        mix_input_dict = dataset.get_data_info(index)
        if self.pre_transform4orig is not None and self.pre_transform4orig_prob > np.random.rand():
            input_dict = self.pre_transform4orig(input_dict)
        if self.pre_transform is not None:
            mix_input_dict.update({'dataset': dataset})
            mix_results = self.pre_transform(mix_input_dict)
            mix_results.pop('dataset')
        input_dict = self.angleselective_transform(input_dict, mix_results)
        return input_dict

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(instance_classes={self.instance_classes}, "
                f"swap_ratio={self.swap_ratio}, rotate_paste_ratio={self.rotate_paste_ratio}, "
                f"pre_transform={self.pre_transform}, prob={self.prob})")

@TRANSFORMS.register_module()
class SimulatedMatterAccumulation(BaseTransform):
    def __init__(self, rho=0.3, h_range=(0.05, 0.3), gamma_range=(0.3, 1.0), prob=1.0, separately_output=False, protect_labels=None):
        self.rho = float(rho)
        self.h1, self.h2 = float(h_range[0]), float(h_range[1])
        self.g1, self.g2 = float(gamma_range[0]), float(gamma_range[1])
        self.prob = float(prob)
        self.separately_output = separately_output
        self.protect_labels = set(protect_labels or [])

    def transform(self, input_dict: dict) -> dict:
        if np.random.rand() > self.prob:
            return input_dict
        points = input_dict['points']
        N = points.coord.shape[0]
        device = points.coord.device
        keep = torch.ones(N, dtype=torch.bool, device=device)
        labels = input_dict.get('pts_semantic_mask', None)
        if labels is not None and len(self.protect_labels) > 0:
            lab_t = torch.from_numpy(labels).to(device) if isinstance(labels, np.ndarray) else labels.to(device)
            for lid in self.protect_labels:
                keep &= (lab_t != int(lid))
        gm = _ground_mask(points, z_thr=0.2)
        idx_pool = torch.nonzero(keep & (~gm), as_tuple=False).squeeze(1)
        if idx_pool.numel() == 0:
            idx_pool = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if idx_pool.numel() == 0:
            return input_dict
        k = max(1, int(round(self.rho * idx_pool.numel())))
        perm = torch.randperm(idx_pool.numel(), device=device)[:k]
        idx = idx_pool[perm]
        z = points.tensor[idx, 2].abs().clamp_(0, 1)
        h = torch.empty(k, device=device).uniform_(self.h1, self.h2) * (0.6 + 0.4 * z)
        g = torch.empty(k, device=device).uniform_(self.g1, self.g2) * (0.7 + 0.3 * z)
        pts = points.tensor
        pts[idx, 2] = pts[idx, 2] + h
        if pts.size(1) > 3:
            pts[idx, -1] = pts[idx, -1] * g
        points.tensor = pts
        if self.separately_output:
            input_dict['points_aug'] = points
            if 'pts_semantic_mask' in input_dict:
                input_dict['pts_semantic_mask_aug'] = input_dict['pts_semantic_mask']
        else:
            input_dict['points'] = points
        return input_dict

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(rho={self.rho}, h_range=({self.h1}, {self.h2}), "
                f"gamma_range=({self.g1}, {self.g2}), prob={self.prob}, protect_labels={sorted(self.protect_labels)})")

@TRANSFORMS.register_module()
class SimulatedFuzzyRecognition(BaseTransform):
    def __init__(self,
                 alpha=0.015,                  # softened default
                 p=0.06,                       # softened default
                 ignore_index=19,
                 prob=1.0,
                 separately_output=False,
                 protect_labels=None,
                 min_keep_range=0.0,
                 use_exp_decay=True,
                 # NEW: soften & smarter gating
                 partial_ignore_ratio=0.5,     # only a subset of candidates are ignored
                 gate_by_density=True,         # protect sparse/boundary regions
                 gate_voxel_size=1.0,
                 gate_quantile=0.25):
        self.alpha = float(alpha)
        self.p = float(p)
        self.ignore_index = int(ignore_index)
        self.prob = float(prob)
        self.separately_output = separately_output
        self.protect_labels = set(protect_labels or [])
        self.min_keep_range = float(min_keep_range)
        self.use_exp_decay = bool(use_exp_decay)
        self.partial_ignore_ratio = float(partial_ignore_ratio)
        self.gate_by_density = bool(gate_by_density)
        self.gate_voxel_size = float(gate_voxel_size)
        self.gate_quantile = float(gate_quantile)

    def transform(self, input_dict: dict) -> dict:
        if np.random.rand() > self.prob:
            return input_dict
        if 'pts_semantic_mask' not in input_dict:
            return input_dict

        points = input_dict['points']
        labels = input_dict['pts_semantic_mask']
        coord = points.coord
        I = points.tensor[:, -1]
        r = torch.linalg.norm(coord, dim=1)

        Ie = I * torch.exp(-self.alpha * r) if self.use_exp_decay else I
        tau = torch.quantile(Ie, self.p)
        mask = Ie < tau
        if self.min_keep_range > 0:
            mask &= (r > self.min_keep_range)

        device = coord.device
        if len(self.protect_labels) > 0:
            lab_t = torch.from_numpy(labels).to(device) if isinstance(labels, np.ndarray) else labels.to(device)
            prot = torch.zeros_like(mask)
            for lid in self.protect_labels:
                prot |= (lab_t == int(lid))
            mask &= (~prot)

        # --- density/boundary gating ---
        if self.gate_by_density and mask.any():
            gsz = max(self.gate_voxel_size, 1e-6)
            g = torch.floor(coord / gsz).long()
            key = g[:, 0] * 73856093 ^ g[:, 1] * 19349663 ^ g[:, 2] * 83492791
            uniq, inv = torch.unique(key, return_inverse=True)
            counts = torch.bincount(inv, minlength=uniq.numel()).to(device)
            dens = counts[inv].float()
            thr = torch.quantile(dens, self.gate_quantile)
            boundary_sparse = dens <= thr
            mask &= (~boundary_sparse)

        # --- partial ignore ---
        if self.partial_ignore_ratio < 1.0 and mask.any():
            m_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            k = int(round(self.partial_ignore_ratio * m_idx.numel()))
            if 0 <= k < m_idx.numel():
                perm = torch.randperm(m_idx.numel(), device=device)
                keep_idx = m_idx[perm[k:]]
                mask[keep_idx] = False

        if isinstance(labels, np.ndarray):
            lab_mod = labels.copy(); lab_mod[mask.cpu().numpy()] = self.ignore_index
            input_dict['pts_semantic_mask'] = lab_mod
        else:
            lab = labels.clone(); lab[mask] = self.ignore_index
            input_dict['pts_semantic_mask'] = lab

        if self.separately_output:
            input_dict['points_aug'] = input_dict['points']
            input_dict['pts_semantic_mask_aug'] = input_dict['pts_semantic_mask']
        return input_dict

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(alpha={self.alpha}, p={self.p}, ignore_index={self.ignore_index}, "
                f"prob={self.prob}, protect_labels={sorted(self.protect_labels)}, min_keep_range={self.min_keep_range}, "
                f"use_exp_decay={self.use_exp_decay}, partial_ignore_ratio={self.partial_ignore_ratio}, "
                f"gate_by_density={self.gate_by_density}, gate_voxel_size={self.gate_voxel_size}, gate_quantile={self.gate_quantile})")

@TRANSFORMS.register_module()
class PhysicsInspiredAdverseGeometrySimulation(BaseTransform):
    def __init__(self, beta1=0.3, beta2=0.5, psi1_cfg=None, psi2_cfg=None, separately_output=False):
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.psi1 = SimulatedMatterAccumulation(**(psi1_cfg or {}))
        self.psi2 = SimulatedFuzzyRecognition(**(psi2_cfg or {}))
        self.separately_output = separately_output

    def transform(self, input_dict: dict) -> dict:
        out = dict(input_dict)
        if np.random.rand() < self.beta1:
            out = self.psi1.transform(out)
        if np.random.rand() < self.beta2:
            out = self.psi2.transform(out)
        if self.separately_output:
            out.setdefault('points_aug', out.get('points'))
            if 'pts_semantic_mask' in out:
                out.setdefault('pts_semantic_mask_aug', out.get('pts_semantic_mask'))
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(beta1={self.beta1}, beta2={self.beta2}, psi1={self.psi1}, psi2={self.psi2})"