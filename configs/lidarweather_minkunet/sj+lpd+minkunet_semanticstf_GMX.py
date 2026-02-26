_base_ = [
    '../_base_/datasets/semanticstf.py', '../_base_/models/minkunet.py',
    '../_base_/default_runtime.py'
]

# ==============================================
# GMX + Learnable Policy Drop + MinkUNet (SemanticSTF)
# ==============================================

learnable_drop = True

# --- Dataset ---
dataset_type = 'SemanticSTFDataset'
# NOTE: 리눅스는 대소문자 구분. 실제 폴더명과 동일하게 맞추세요.
data_root = '/home/vip/harry/LiDARWeather/data/sets/SemanticSTF'

class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
]

# SemanticSTF 라벨 정의(예: 0..20, 20은 invalid). 19-class 학습/평가(unlabeled=ignore)
labels_map = {
    0: 19,  # unlabeled -> ignore
    1: 0,   # car
    2: 1,   # bicycle
    3: 2,   # motorcycle
    4: 3,   # truck
    5: 4,   # other-vehicle  (본 설정에서는 bus로 병합/매핑)
    6: 5,   # person
    7: 6,   # bicyclist
    8: 7,   # motorcyclist
    9: 8,   # road
    10: 9,  # parking
    11: 10, # sidewalk
    12: 11, # other-ground
    13: 12, # building
    14: 13, # fence
    15: 14, # vegetation
    16: 15, # trunk
    17: 16, # terrain
    18: 17, # pole
    19: 18, # traffic-sign
    20: 19  # invalid -> ignore
}

metainfo = dict(
    classes=class_names,
    seg_label_mapping=labels_map,
    max_label=20
)

input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

# --- Pipelines ---
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4,
         backend_args=backend_args, norm_intensity255=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_seg_3d=True,
         seg_3d_dtype='np.int32', seg_offset=2**16, dataset_type='semantickitti', backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomChoice',
        transforms=[
            dict(
                type='DepthSelectiveJitter',
                num_areas=[5, 6, 7, 8],
                pitch_angles=[-25, 3],
                pre_transform=[
                    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4,
                         backend_args=backend_args, norm_intensity255=True),
                    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_seg_3d=True,
                         seg_3d_dtype='np.int32', seg_offset=2**16, dataset_type='semantickitti', backend_args=backend_args),
                    dict(type='PointSegClassMapping'),
                    dict(type='RandomJitterPoints', jitter_std=[0.01, 0.01, 0.01],
                         clip_range=[-0.05, 0.05], reflectance_noise=True)
                ],
                pre_transform4orig=[
                    dict(type='RandomRangeJitterPoints', jitter_std=0.01,
                         clip_range=[-0.05, 0.05], reflectance_noise=True)
                ],
                pre_transform4orig_prob=0.5,
                prob=0.5
            ),
            dict(
                type='AngleSelectiveJitter',
                instance_classes=[0, 1, 2, 3, 4, 5, 6, 7],
                swap_ratio=0.5,
                rotate_paste_ratio=1.0,
                pre_transform=[
                    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4,
                         backend_args=backend_args, norm_intensity255=True),
                    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_seg_3d=True,
                         seg_3d_dtype='np.int32', seg_offset=2**16, dataset_type='semantickitti', backend_args=backend_args),
                    dict(type='PointSegClassMapping'),
                    dict(type='RandomJitterPoints', jitter_std=[0.01, 0.01, 0.01],
                         clip_range=[-0.05, 0.05], reflectance_noise=True)
                ],
                pre_transform4orig=[
                    dict(type='RandomRangeJitterPoints', jitter_std=0.01,
                         clip_range=[-0.05, 0.05], reflectance_noise=True)
                ],
                pre_transform4orig_prob=0.5,
                prob=0.5
            )
        ],
        prob=[0.5, 0.5]
    ),
    dict(type='GlobalRotScaleTrans', rot_range=[0., 6.28318531],
         scale_ratio_range=[0.95, 1.05], translation_std=[0, 0, 0]),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

# TEST 파이프라인
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=4,
         backend_args=backend_args, norm_intensity255=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_seg_3d=True,
         seg_3d_dtype='np.int32', seg_offset=2**16, dataset_type='semantickitti', backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

# --- Dataloaders ---
# train pkl이 루트에 있고, 실제 데이터는 train/ 하위에 있다면 아래 data_prefix를 사용하세요.
train_dataloader = dict(
    sampler=dict(seed=0),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,                          # pkl 위치(루트)
        ann_file='semanticstf_infos_train.pkl',       # 루트의 train pkl
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=19,
        data_prefix=dict(pts='train/velodyne', pts_semantic_mask='train/labels'),
        backend_args=backend_args)
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root, 
        ann_file='semanticstf_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=19,
        test_mode=True,
        data_prefix=dict(pts='val', pts_semantic_mask='val'),
        backend_args=backend_args)
)

# --- Model ---
model = dict(
    type='MinkUNetWeatherDropper',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor', voxel=True, voxel_type='minkunet',
        batch_first=False, max_voxels=80000,
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=[-100, -100, -20, 100, 100, 20],
            voxel_size=[0.05, 0.05, 0.05],
            max_voxels=(-1, -1)
        )
    ),
    backbone=dict(
        type='MinkUNetBackbone',
        in_channels=4,
        num_stages=4,
        base_channels=32,
        encoder_channels=[32, 64, 128, 256],
        encoder_blocks=[2, 2, 2, 2],
        decoder_channels=[256, 128, 96, 96],
        decoder_blocks=[2, 2, 2, 2],
        block_type='basic',
        sparseconv_backend='torchsparse'
    ),
    decode_head=dict(
        type='MinkUNetHead',
        channels=96,
        num_classes=19,
        batch_first=False,
        dropout_ratio=0,
        loss_decode=dict(type='mmdet.CrossEntropyLoss', avg_non_ignore=True),
        ignore_index=19
    ),
    train_cfg=dict(learnable_drop=learnable_drop),
    test_cfg=dict()
)

# --- Optim & Schedule ---
lr = 0.24
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='SGD', lr=lr, weight_decay=0.0001, momentum=0.9, nesterov=True)
)
optimizer_config = dict(grad_clip=dict(max_norm=100, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.008, by_epoch=False, begin=0, end=125),
    dict(type='CosineAnnealingLR', begin=0, T_max=15, by_epoch=True, eta_min=1e-5, convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# --- Hooks & DDP ---
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='miou', rule='greater')
)
custom_hooks = [dict(type='mmdet3d.engine.hooks.policy_target_hook.PolicyTargetHook')] if learnable_drop else []

randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)
env_cfg = dict(cudnn_benchmark=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=True, broadcast_buffers=False)

ddp_cfg = dict(find_unused_parameters=True, broadcast_buffers=False)