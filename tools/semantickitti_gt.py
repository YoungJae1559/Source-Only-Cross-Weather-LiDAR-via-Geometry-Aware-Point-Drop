import argparse
import os
import os.path as osp
import numpy as np
import torch

from mmengine.config import Config
from mmdet3d.registry import DATASETS, VISUALIZERS
from mmdet3d.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file for SemanticKITTI')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--max-num', type=int, default=-1)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def get_dataset_cfg(cfg, split):
    if split == 'train':
        dl_cfg = cfg.train_dataloader
    else:
        dl_cfg = cfg.val_dataloader if hasattr(cfg, 'val_dataloader') else cfg.test_dataloader
    dataset_cfg = dl_cfg['dataset'] if isinstance(dl_cfg, dict) else dl_cfg[0]['dataset']
    dataset_cfg = dataset_cfg.copy()
    dataset_cfg['test_mode'] = (split != 'train')
    return dataset_cfg


def build_dataset(cfg, split):
    return DATASETS.build(get_dataset_cfg(cfg, split))


def get_metainfo_dict(data_sample):
    if hasattr(data_sample, 'metainfo'):
        try:
            meta = data_sample.metainfo
            if isinstance(meta, dict):
                return meta
        except Exception:
            pass
    return {}


def load_semkitti_points(meta):
    lidar_path = meta.get('lidar_path') or meta.get('pts_filename') or meta.get('point_path') or meta.get('pcd_path')
    if lidar_path is None:
        raise RuntimeError('Cannot find lidar path in data_sample.metainfo')
    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    pts = torch.from_numpy(pts)
    return pts


def extract_points_from_inputs_or_file(inputs, data_sample):
    if isinstance(inputs, dict) and 'points' in inputs and inputs['points'] is not None:
        return inputs['points']
    meta = get_metainfo_dict(data_sample)
    pts = load_semkitti_points(meta)
    return pts


def make_stem(data_sample, idx):
    meta = get_metainfo_dict(data_sample)
    for k in ['lidar_path', 'pts_filename', 'point_path', 'pcd_path', 'sample_idx']:
        v = meta.get(k, None)
        if v is None:
            continue
        if isinstance(v, str):
            b = osp.splitext(osp.basename(v))[0]
            if b:
                return b
        return str(v)
    return f'{idx:06d}'


def ensure_palette(dataset_meta, max_label):
    if not isinstance(dataset_meta, dict):
        return
    palette = dataset_meta.get('palette', None)
    if palette is None:
        return
    pal = list(palette)
    while len(pal) <= max_label:
        pal.append([0, 0, 0])
    dataset_meta['palette'] = pal
    classes = dataset_meta.get('classes', None)
    if classes is not None:
        cls = list(classes)
        while len(cls) < len(pal):
            cls.append('ignore')
        dataset_meta['classes'] = cls


def get_max_label_from_pred_pts_seg(pred_pts_seg):
    labels = None
    if hasattr(pred_pts_seg, 'pts_semantic_mask'):
        labels = pred_pts_seg.pts_semantic_mask
    elif hasattr(pred_pts_seg, 'semantic_mask'):
        labels = pred_pts_seg.semantic_mask
    elif isinstance(pred_pts_seg, dict):
        if 'pts_semantic_mask' in pred_pts_seg:
            labels = pred_pts_seg['pts_semantic_mask']
        elif 'semantic_mask' in pred_pts_seg:
            labels = pred_pts_seg['semantic_mask']

    if labels is None:
        return None

    if isinstance(labels, torch.Tensor):
        if labels.numel() == 0:
            return None
        return int(labels.max().item())

    arr = np.asarray(labels)
    if arr.size == 0:
        return None
    return int(arr.max())


def main():
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg, args.split)

    visualizer = VISUALIZERS.build(cfg.visualizer)
    if hasattr(dataset, 'metainfo'):
        visualizer.dataset_meta = dataset.metainfo

    os.makedirs(args.out_dir, exist_ok=True)

    max_num = len(dataset) if args.max_num < 0 or args.max_num > len(dataset) else args.max_num

    for idx in range(max_num):
        data = dataset[idx]

        if 'inputs' in data:
            inputs = data['inputs']
        elif 'data_input' in data:
            inputs = data['data_input']
        else:
            raise RuntimeError('inputs or data_input not found in sample')

        ds = data.get('data_samples', data.get('data_sample', None))
        if ds is None:
            raise RuntimeError('data_samples or data_sample not found in sample')

        data_sample = ds[0] if isinstance(ds, list) else ds
        pred_sample = data_sample.clone()

        if hasattr(pred_sample, 'gt_pts_seg'):
            pred_sample.pred_pts_seg = pred_sample.gt_pts_seg
            del pred_sample.gt_pts_seg
        elif hasattr(pred_sample, 'gt_semantic_seg'):
            pred_sample.pred_pts_seg = pred_sample.gt_semantic_seg
            del pred_sample.gt_semantic_seg
        else:
            raise RuntimeError('gt_pts_seg or gt_semantic_seg not found in data_sample')

        max_label = get_max_label_from_pred_pts_seg(pred_sample.pred_pts_seg)
        if max_label is not None:
            ensure_palette(visualizer.dataset_meta, max_label)

        pts = extract_points_from_inputs_or_file(inputs, data_sample)
        data_input = {'points': pts} if not (isinstance(inputs, dict) and 'points' in inputs) else inputs

        pred_sample = pred_sample.numpy()

        stem = make_stem(data_sample, idx)
        save_path = osp.join(args.out_dir, f'{stem}.png')

        visualizer.add_datasample(
            name=f'semkitti_gt_{stem}',
            data_input=data_input,
            data_sample=pred_sample,
            show=args.show,
            wait_time=0,
            out_file=save_path,
            o3d_save_path=save_path,
            vis_task='lidar_seg'
        )


if __name__ == '__main__':
    main()
