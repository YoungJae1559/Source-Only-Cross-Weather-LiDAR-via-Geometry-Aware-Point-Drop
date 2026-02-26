import argparse
import os
import os.path as osp

from mmengine.config import Config
from mmdet3d.registry import DATASETS, VISUALIZERS
from mmdet3d.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file for SemanticSTF')
    parser.add_argument('--out-dir', required=True, help='directory to save GT visualizations')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--max-num', type=int, default=-1)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def build_dataset_from_cfg(cfg, split):
    if split == 'val':
        dl_cfg = cfg.test_dataloader
    else:
        dl_cfg = cfg.train_dataloader

    if isinstance(dl_cfg, dict):
        dataset_cfg = dl_cfg['dataset']
    else:
        dataset_cfg = dl_cfg[0]['dataset']

    dataset_cfg['test_mode'] = (split == 'val')

    dataset = DATASETS.build(dataset_cfg)
    return dataset


def main():
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset_from_cfg(cfg, args.split)

    visualizer = VISUALIZERS.build(cfg.visualizer)
    if hasattr(dataset, 'metainfo'):
        visualizer.dataset_meta = dataset.metainfo

    os.makedirs(args.out_dir, exist_ok=True)

    if args.max_num < 0 or args.max_num > len(dataset):
        max_num = len(dataset)
    else:
        max_num = args.max_num

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

        if isinstance(ds, list):
            data_sample = ds[0]
        else:
            data_sample = ds

        pred_sample = data_sample.clone()

        if hasattr(pred_sample, 'gt_pts_seg'):
            pred_sample.pred_pts_seg = pred_sample.gt_pts_seg
        elif hasattr(pred_sample, 'gt_semantic_seg'):
            pred_sample.pred_pts_seg = pred_sample.gt_semantic_seg
        else:
            raise RuntimeError('gt_pts_seg or gt_semantic_seg not found in data_sample')

        pred_sample = pred_sample.numpy()

        o3d_save_path = osp.join(args.out_dir, f'{idx:06d}.png')

        visualizer.add_datasample(
            name=f'semstf_gt_{idx:06d}',
            data_input=inputs,
            data_sample=pred_sample,
            show=args.show,          # 창 띄우고 싶으면 --show, 아니면 옵션 빼고 실행
            wait_time=0,
            out_file=None,           # lidar_seg에서는 보통 의미 없음
            o3d_save_path=o3d_save_path,
            vis_task='lidar_seg'
        )


if __name__ == '__main__':
    main()