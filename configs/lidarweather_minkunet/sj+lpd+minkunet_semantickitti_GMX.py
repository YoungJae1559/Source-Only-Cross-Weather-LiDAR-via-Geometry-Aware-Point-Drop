_base_ = [
    '../_base_/datasets/semantickitti.py', '../_base_/models/minkunet.py',
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
USE_PSI1 = False
USE_PSI2 = False  # ← 여기만 True로 바꾸면 PSI2 켜짐

learnable_drop = True

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
    broadcast_buffers=False,
)

# Dataset/meta
dataset_type = 'SemanticKittiDataset'
data_root = '/home/vip/harry/LiDARWeather/data/sets/semantickitti'
class_names = [
    'car','bicycle','motorcycle','truck','bus','person','bicyclist',
    'motorcyclist','road','parking','sidewalk','other-ground','building',
    'fence','vegetation','trunck','terrian','pole','traffic-sign'
]
labels_map = {
    0:19,1:19,10:0,11:1,13:4,15:2,16:4,18:3,20:4,21:19,22:19,23:19,30:5,31:6,32:7,
    40:8,44:9,48:10,49:11,50:12,51:13,52:19,60:8,70:14,71:15,72:16,80:17,81:18,99:19,
    252:0,253:6,254:5,255:7,256:4,257:4,258:3,259:4
}
metainfo = dict(classes=class_names, seg_label_mapping=labels_map, max_label=259)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

# ---------------- Train pipeline ----------------
# Load -> Labels -> Mapping -> (DSJ|ASJ) -> GlobalRotScale -> Ψ1 -> Ψ2(옵션, 항상 마지막) -> Pack
_train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_seg_3d=True,
         seg_3d_dtype='np.int32', seg_offset=2**16, dataset_type='semantickitti'),
    dict(type='PointSegClassMapping'),
]

# 1) DSJ or ASJ
_train_pipeline += [
    dict(
        type='RandomChoice',
        transforms=[
            dict(
                type='DepthSelectiveJitter',
                num_areas=[5,6,7,8],
                pitch_angles=[-25,3],
                pre_transform=[
                    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
                    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_seg_3d=True,
                         seg_3d_dtype='np.int32', seg_offset=2**16, dataset_type='semantickitti'),
                    dict(type='PointSegClassMapping'),
                    dict(type='RandomJitterPoints', jitter_std=[0.01,0.01,0.01], clip_range=[-0.05,0.05],
                         reflectance_noise=True),
                ],
                pre_transform4orig=[
                    dict(type='RandomRangeJitterPoints', jitter_std=0.01, clip_range=[-0.05,0.05], reflectance_noise=True),
                ],
                pre_transform4orig_prob=0.5,
                prob=0.5,
            ),
            dict(
                type='AngleSelectiveJitter',
                instance_classes=[0,1,2,3,4,5,6,7],
                swap_ratio=0.5,
                rotate_paste_ratio=1.0,
                pre_transform=[
                    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
                    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_seg_3d=True,
                         seg_3d_dtype='np.int32', seg_offset=2**16, dataset_type='semantickitti'),
                    dict(type='PointSegClassMapping'),
                    dict(type='RandomJitterPoints', jitter_std=[0.01,0.01,0.01], clip_range=[-0.05,0.05],
                         reflectance_noise=True),
                ],
                pre_transform4orig=[
                    dict(type='RandomRangeJitterPoints', jitter_std=0.01, clip_range=[-0.05,0.05], reflectance_noise=True),
                ],
                pre_transform4orig_prob=0.5,
                prob=0.5,
            ),
        ],
        prob=[0.5,0.5],
    ),
]

# 2) GlobalRotScale
_train_pipeline += [
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0,0,0],
    ),
]

# 3) Ψ1 (SimulatedMatterAccumulation)
if USE_PSI1:
    _train_pipeline += [
        dict(
            type='RandomChoice',
            transforms=[
                [dict(
                    type='SimulatedMatterAccumulation',
                    rho=0.08,
                    h_range=[0.01, 0.06],
                    gamma_range=[0.75, 1.0],
                    prob=1.0,
                )],
                []
            ],
            prob=[0.20, 0.80],
        )
    ]

# 4) Ψ2 (SimulatedFuzzyRecognition) 항상 마지막
if USE_PSI2:
    _train_pipeline += [
        dict(
            type='RandomChoice',
            transforms=[
                [dict(
                    type='SimulatedFuzzyRecognition',
                    mode='exp_decay',   # r에 따라 I를 exp(-alpha*r)로 감쇠
                    alpha=0.04,
                    p=0.20,             # 하위 20%를 ignore로 마스킹
                    ignore_index=19,
                    prob=1.0,
                )],
                []
            ],
            prob=[0.20, 0.80],          # 적용 빈도 토글
        )
    ]

# 5) Pack
_train_pipeline += [
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
train_pipeline = _train_pipeline

# ---------------- Test/Val pipeline ----------------
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_seg_3d=True,
         seg_3d_dtype='np.int32', seg_offset=2**16, dataset_type='semantickitti', backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

# dataloaders
test_dataloader = dict(
    batch_size=1, num_workers=1, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file='semantickitti_infos_val.pkl',
        pipeline=test_pipeline, metainfo=metainfo, modality=input_modality, ignore_index=19, test_mode=True,
        backend_args=backend_args))

# model
model = dict(
    type='MinkUNetWeatherDropper',
    use_gmx_lite=USE_GMX,
    n_observations=4 if USE_GMX else 3,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor', voxel=True, voxel_type='minkunet', batch_first=False, max_voxels=80000,
        voxel_layer=dict(max_num_points=-1, point_cloud_range=[-100,-100,-20, 100,100,20],
                         voxel_size=[0.05,0.05,0.05], max_voxels=(-1,-1))),
    backbone=dict(
        type='MinkUNetBackbone', in_channels=4, num_stages=4, base_channels=32,
        encoder_channels=[32,64,128,256], encoder_blocks=[2,2,2,2],
        decoder_channels=[256,128,96,96], decoder_blocks=[2,2,2,2],
        block_type='basic', sparseconv_backend='torchsparse'),
    decode_head=dict(
        type='MinkUNetHead', channels=96, num_classes=19, batch_first=False, dropout_ratio=0,
        loss_decode=dict(type='mmdet.CrossEntropyLoss', avg_non_ignore=True), ignore_index=19),
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
        dqn_drop_bins=[0.1,0.3,0.5],
        region_voxel_size=1.0,
        gmx_k=20,
        gt_fg_label_min=1,
        gt_penalty_lambda=0.5,
    ),
    test_cfg=dict()
)

train_dataloader = dict(sampler=dict(seed=1), dataset=dict(pipeline=train_pipeline))

# optim & sched
lr = 0.24
optim_wrapper = dict(
    type='AmpOptimWrapper', loss_scale='dynamic',
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

# hooks
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, save_best='miou', rule='greater'))
if learnable_drop:
    custom_hooks = [dict(type='mmdet3d.engine.hooks.policy_target_hook.PolicyTargetHook')]

randomness = dict(seed=1, deterministic=False, diff_rank_seed=True)
env_cfg = dict(cudnn_benchmark=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')