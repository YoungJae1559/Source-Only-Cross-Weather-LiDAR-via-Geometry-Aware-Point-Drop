import argparse
import os
import os.path as osp
from copy import deepcopy

import numpy as np
import open3d as o3d
import torch
from mmengine.config import Config
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Interactive visualize original vs snow simulated SemanticKITTI')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--out-dir', required=True, help='directory to save visualizations')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--max-num', type=int, default=1)
    parser.add_argument('--start-idx', type=int, default=0)
    return parser.parse_args()


def fix_load_dim(dataset_cfg):
    pipeline = dataset_cfg['pipeline']
    for t in pipeline:
        if t.get('type', '') == 'LoadPointsFromFile':
            t['load_dim'], t['use_dim'] = 4, [0, 1, 2, 3]
    return dataset_cfg


def insert_snow_into_pipeline(dataset_cfg):
    dataset_cfg = fix_load_dim(dataset_cfg)
    pipeline = dataset_cfg['pipeline']
    snow_transform = dict(
        type='SimulatedMatterAccumulation',
        rho=0.60,
        h_range=[0.04, 0.16],
        gamma_range=[0.4, 1.0],
        prob=1.0,
        separately_output=False
    )
    pack_idx = None
    for i, t in enumerate(pipeline):
        if t.get('type', '') == 'Pack3DDetInputs':
            pack_idx = i
            break
    if pack_idx is None:
        raise RuntimeError('Pack3DDetInputs not found in pipeline.')
    pipeline.insert(pack_idx, snow_transform)
    dataset_cfg['pipeline'] = pipeline
    return dataset_cfg


def build_datasets(cfg, split):
    if split == 'val':
        base_cfg = deepcopy(cfg.test_dataloader['dataset'])
    else:
        base_cfg = deepcopy(cfg.train_dataloader['dataset'])
        base_cfg['test_mode'] = False
    base_cfg = fix_load_dim(base_cfg)
    orig_cfg = deepcopy(base_cfg)
    snow_cfg = insert_snow_into_pipeline(deepcopy(base_cfg))
    orig_ds = DATASETS.build(orig_cfg)
    snow_ds = DATASETS.build(snow_cfg)
    return orig_ds, snow_ds, orig_cfg


def make_pcd(data, palette, ignore_index, num_classes):
    inputs = data['inputs']
    ds = data['data_samples']
    if isinstance(ds, list):
        ds = ds[0]
    p = inputs['points']
    pts = p.tensor.cpu().numpy() if hasattr(p, 'tensor') else p.cpu().numpy()
    if not hasattr(ds, 'gt_pts_seg'):
        raise RuntimeError('gt_pts_seg not found in data_sample')
    seg = ds.gt_pts_seg.pts_semantic_mask
    if isinstance(seg, torch.Tensor):
        seg = seg.cpu().numpy()
    labels = seg.astype(np.int32)
    mask_valid = labels != ignore_index
    labels_clamped = np.clip(labels, 0, num_classes - 1)
    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
    colors[mask_valid] = palette[labels_clamped[mask_valid]]
    xyz = pts[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _window_size_from_cam(cam_params, default_w=1920, default_h=1080):
    if cam_params is None:
        return default_w, default_h
    intr = cam_params.intrinsic
    w = int(getattr(intr, 'width', default_w))
    h = int(getattr(intr, 'height', default_h))
    return w, h


def interactive_lock_view(pcd, window_name, cam_params=None):
    w, h = _window_size_from_cam(cam_params)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=w, height=h, visible=True)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 3.0

    vc = vis.get_view_control()
    if cam_params is not None:
        try:
            vc.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        except TypeError:
            vc.convert_from_pinhole_camera_parameters(cam_params)

    holder = {}

    def lock_and_close(v):
        holder['params'] = v.get_view_control().convert_to_pinhole_camera_parameters()
        v.close()
        return False

    vis.register_key_callback(ord('S'), lock_and_close)
    vis.register_key_callback(ord('s'), lock_and_close)
    print('첫 프레임 original에서 마우스로 뷰 조정 후 S를 누르면 그 뷰가 고정됩니다.')
    vis.run()
    vis.destroy_window()
    return holder.get('params', cam_params)


def save_with_fixed_view(pcd, out_path, window_name, cam_params):
    w, h = _window_size_from_cam(cam_params)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=w, height=h, visible=True)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 3.0

    vc = vis.get_view_control()
    try:
        ok = vc.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
    except TypeError:
        ok = vc.convert_from_pinhole_camera_parameters(cam_params)

    vis.poll_events()
    vis.update_renderer()
    vis.poll_events()
    vis.update_renderer()

    if ok is False:
        print('경고: camera params 적용이 실패해서 기본 뷰로 저장됐을 수 있습니다.')

    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()


def main():
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    cam_file = osp.join(args.out_dir, 'camera.json')
    cam_params = o3d.io.read_pinhole_camera_parameters(cam_file) if osp.exists(cam_file) else None

    orig_ds, snow_ds, dataset_cfg = build_datasets(cfg, args.split)
    meta = getattr(orig_ds, 'metainfo', {})
    palette = np.array(meta.get('palette', []), dtype=np.float32) / 255.0
    if palette.size == 0:
        num_classes = 19
        rng = np.random.RandomState(0)
        palette = rng.randint(0, 255, size=(num_classes, 3)).astype(np.float32) / 255.0
    num_classes = palette.shape[0]
    ignore_index = dataset_cfg.get('ignore_index', 19)

    max_num = len(orig_ds) if args.max_num < 0 or args.max_num > len(orig_ds) else args.max_num
    start = max(0, min(args.start_idx, len(orig_ds) - 1))
    end = min(start + max_num, len(orig_ds))

    for i, idx in enumerate(range(start, end)):
        data_orig = orig_ds[idx]
        data_snow = snow_ds[idx]

        pcd0 = make_pcd(data_orig, palette, ignore_index, num_classes)
        pcd1 = make_pcd(data_snow, palette, ignore_index, num_classes)

        out0 = osp.join(args.out_dir, f'{idx:06d}_orig.png')
        out1 = osp.join(args.out_dir, f'{idx:06d}_snow.png')

        if i == 0:
            cam_params = interactive_lock_view(pcd0, f'orig_lock_{idx:06d}', cam_params)
            if cam_params is None:
                raise RuntimeError('S를 눌러 뷰를 고정해야 합니다.')
            o3d.io.write_pinhole_camera_parameters(cam_file, cam_params)

            save_with_fixed_view(pcd0, out0, f'orig_{idx:06d}', cam_params)
            save_with_fixed_view(pcd1, out1, f'snow_{idx:06d}', cam_params)
            continue

        if cam_params is None:
            raise RuntimeError('camera params가 없습니다. 첫 프레임에서 S로 뷰를 고정하세요.')

        save_with_fixed_view(pcd0, out0, f'orig_{idx:06d}', cam_params)
        save_with_fixed_view(pcd1, out1, f'snow_{idx:06d}', cam_params)


if __name__ == '__main__':
    main()
