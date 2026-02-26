_base_ = [
    '../_base_/datasets/synlidar.py', '../_base_/models/minkunet.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmdet3d.datasets.transforms',
        'mmdet3d.datasets.transforms_3d',
        'projects.lidarweather.transforms_3d',
    ],
    allow_failed_imports=True,
)

USE_GMX = True
USE_DSJ_ASJ = True
USE_PSI1 = False
USE_PSI2 = False

#PSI1_TOGGLE = 0.20
#PSI1_CFG = dict(rho=0.08, h_range=[0.01, 0.06], gamma_range=[0.75, 1.0], prob=1.0)

PSI1_TOGGLE = 0.12
PSI1_CFG = dict(rho=0.05, h_range=[0.005, 0.03], gamma_range=[0.9, 1.0], prob=1.0)

PSI2_TOGGLE = 0.20
PSI2_CFG = dict(mode='exp_decay', alpha=0.04, p=0.20, ignore_index=19, prob=1.0)

learnable_drop = True

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
    broadcast_buffers=False,
)

dataset_type = 'SynLiDARDataset'
data_root = '/home/vip/harry/LiDARWeather/data/sets/SynLiDAR'
class_names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign'
]
labels_map = {
    0: 19,
    1: 0,
    2: 3,
    3: 3,
    4: 4,
    5: 1,
    6: 2,
    7: 4,
    8: 8,
    9: 10,
    10: 9,
    11: 11,
    12: 5,
    13: 5,
    14: 5,
    15: 5,
    16: 6,
    17: 7,
    18: 12,
    19: 19,
    20: 14,
    21: 15,
    22: 16,
    23: 18,
    24: 17,
    25: 19,
    26: 13,
    27: 19,
    28: 19,
    29: 19,
    30: 19,
    31: 19,
    32: 19,
}
metainfo = dict(classes=class_names, seg_label_mapping=labels_map, max_label=259)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

_train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity_minmax=True
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint32',
        seg_offset=2**16,
        dataset_type='semantickitti'
    ),
    dict(type='PointSegClassMapping'),
]

if USE_DSJ_ASJ:
    _train_pipeline += [
        dict(
            type='RandomChoice',
            transforms=[
                dict(
                    type='DepthSelectiveJitter',
                    num_areas=[5, 6, 7, 8],
                    pitch_angles=[-25, 3],
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4,
                            norm_intensity_minmax=True
                        ),
                        dict(
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.uint32',
                            seg_offset=2**16,
                            dataset_type='semantickitti'
                        ),
                        dict(type='PointSegClassMapping'),
                        dict(
                            type='RandomJitterPoints',
                            jitter_std=[0.01, 0.01, 0.01],
                            clip_range=[-0.05, 0.05],
                            reflectance_noise=True
                        ),
                    ],
                    pre_transform4orig=[
                        dict(
                            type='RandomRangeJitterPoints',
                            jitter_std=0.01,
                            clip_range=[-0.05, 0.05],
                            reflectance_noise=True
                        )
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
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4,
                            norm_intensity_minmax=True
                        ),
                        dict(
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.uint32',
                            seg_offset=2**16,
                            dataset_type='semantickitti'
                        ),
                        dict(type='PointSegClassMapping'),
                        dict(
                            type='RandomJitterPoints',
                            jitter_std=[0.01, 0.01, 0.01],
                            clip_range=[-0.05, 0.05],
                            reflectance_noise=True
                        ),
                    ],
                    pre_transform4orig=[
                        dict(
                            type='RandomRangeJitterPoints',
                            jitter_std=0.01,
                            clip_range=[-0.05, 0.05],
                            reflectance_noise=True
                        )
                    ],
                    pre_transform4orig_prob=0.5,
                    prob=0.5
                ),
            ],
            prob=[0.5, 0.5]
        ),
    ]

_train_pipeline += [
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
]

if USE_PSI1:
    _train_pipeline += [
        dict(
            type='RandomChoice',
            transforms=[[dict(type='SimulatedMatterAccumulation', **PSI1_CFG)], []],
            prob=[PSI1_TOGGLE, 1 - PSI1_TOGGLE]
        )
    ]

if USE_PSI2:
    _train_pipeline += [
        dict(
            type='RandomChoice',
            transforms=[[dict(type='SimulatedFuzzyRecognition', **PSI2_CFG)], []],
            prob=[PSI2_TOGGLE, 1 - PSI2_TOGGLE]
        )
    ]

_train_pipeline += [
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_pipeline = _train_pipeline

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity_minmax=True,
        backend_args=backend_args
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args
    ),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True, seed=0),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='synlidar_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=19,
        test_mode=False,
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='synlidar_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=19,
        test_mode=True,
        backend_args=backend_args
    )
)

test_dataloader = val_dataloader

model = dict(
    type='MinkUNetWeatherDropper',
    use_gmx_lite=USE_GMX,
    n_observations=4 if USE_GMX else 3,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='minkunet',
        batch_first=False,
        max_voxels=80000,
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
    train_cfg=dict(
        learnable_drop=learnable_drop,
        use_gmx_lite=USE_GMX,
        dqn_warmup_iters=100,
        dqn_batch_size=32,
        dqn_gamma=0.99,
        dqn_eps_start=0.9,
        dqn_eps_end=0.05,
        dqn_eps_decay=3000,
        dqn_tau=0.01,
        dqn_replay_memory_size=10000,
        dqn_lr=1e-3,
        dqn_weight_decay=0.0,
        dqn_grad_clip=1.0,
        dqn_drop_bins=[0.1, 0.3, 0.5],
        region_voxel_size=1.0,
        gmx_k=16,
        gt_fg_label_min=1,
        gt_penalty_lambda=0.5,
    ),
    test_cfg=dict()
)

lr = 0.24
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='SGD',
        lr=lr,
        weight_decay=0.0001,
        momentum=0.9,
        nesterov=True
    )
)
optimizer_config = dict(grad_clip=dict(max_norm=100, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.008, by_epoch=False, begin=0, end=125),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=15,
        by_epoch=True,
        eta_min=1e-5,
        convert_to_iter_based=True
    )
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='miou', rule='greater')
)
if learnable_drop:
    custom_hooks = [dict(type='mmdet3d.engine.hooks.policy_target_hook.PolicyTargetHook')]

randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)
env_cfg = dict(cudnn_benchmark=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
