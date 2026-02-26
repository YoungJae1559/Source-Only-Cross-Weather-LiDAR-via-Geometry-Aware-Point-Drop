from os import path as osp
from pathlib import Path

import mmengine

fold_split = {
    'train': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'val': [0],
    'test': [0],
}


def absolute_file_paths(directory):
    import os

    files = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            files.append(osp.abspath(osp.join(dirpath, f)))
    files.sort()
    return files


def get_synlidar_info(split, data_root):
    data_root = osp.abspath(data_root)

    data_infos = {'metainfo': {'DATASET': 'SynLiDAR'}, 'data_list': []}

    for seq in fold_split[split]:
        velodyne_dir = osp.join(data_root, 'sequences', f'{seq:02d}', 'velodyne')
        if not osp.isdir(velodyne_dir):
            raise FileNotFoundError(f'{velodyne_dir} not found')

        file_list = absolute_file_paths(velodyne_dir)
        for file_path in file_list:
            fname = osp.basename(file_path)
            stem = osp.splitext(fname)[0]
            data_infos['data_list'].append(
                {
                    'lidar_points': {
                        'lidar_path': osp.join('sequences', f'{seq:02d}', 'velodyne', fname),
                        'num_pts_feats': 4,
                    },
                    'pts_semantic_mask_path': osp.join('sequences', f'{seq:02d}', 'labels', f'{stem}.label'),
                    'sample_id': f'{seq:02d}_{stem}',
                }
            )

    return data_infos


def create_synlidar_info_file(pkl_prefix, save_path, data_root=None):
    print('Generate info.')

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if data_root is None:
        data_root = str(save_path)

    synlidar_infos_train = get_synlidar_info(split='train', data_root=data_root)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'SynLiDAR info train file is saved to {filename}')
    mmengine.dump(synlidar_infos_train, filename)

    synlidar_infos_val = get_synlidar_info(split='val', data_root=data_root)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'SynLiDAR info val file is saved to {filename}')
    mmengine.dump(synlidar_infos_val, filename)

    synlidar_infos_test = get_synlidar_info(split='test', data_root=data_root)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'SynLiDAR info test file is saved to {filename}')
    mmengine.dump(synlidar_infos_test, filename)