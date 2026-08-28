import os
from os import path
import json
from typing import List, Callable, Union, Tuple

from lightning.pytorch import LightningDataModule
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py

from .hdf5 import HDF5Dataset
from .utils import two_stage_sampling
from .transform import AxisScaling
from .base import BaseDataModule


class ShapeNetOccupancyDataModule(BaseDataModule):

    def __init__(self,
                 root_dir: str,

                 ## dataloader params
                 num_workers: int = 4,
                 batch_size: int = 16,
                 eval_batch_size: int = -1,
                 prefetch_factor: int = 2,
                 model_selection_split: str = 'test',
                 evaluation_split: str = 'val',

                 # preprocessing
                 raw_root_dir: str = None,

                 # generation params
                 num_generation_samples: int = -1,
                 shapenet_metadata_path: str = 'assets/shapenet_synset_dict.json',

                 # dataset
                 categories: List[str] = None,
                 return_full_surface: bool = False,
                 num_query_points: int = 4096,
                 repeat: int = 1,
                 pc_size: int = 2048,
                 chunk_size: int = 2000,
                 oversample_ratio: int = 10,
                 use_queries: bool = True,
                 use_full_surface: bool = False,
                 dataset_format: str = 'shapenet',
                 **preprocess_options,
                 ):
        super().__init__()

        self.root_dir = root_dir
        self.eval_grid_size = 128

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size > 0 else batch_size
        self.prefetch_factor = prefetch_factor
        self.model_selection_split = self._validate_eval_split(
            model_selection_split, 'model_selection_split'
        )
        self.evaluation_split = self._validate_eval_split(
            evaluation_split, 'evaluation_split'
        )

        self.raw_root_dir = raw_root_dir
        self.num_generation_samples = num_generation_samples

        self.categories = categories
        self.return_full_surface = return_full_surface
        self.num_query_points = num_query_points
        self.repeat = repeat
        self.pc_size = pc_size
        self.chunk_size = chunk_size
        self.oversample_ratio = oversample_ratio
        self.use_queries = use_queries
        self.use_full_surface = use_full_surface
        if dataset_format.lower() != 'shapenet':
            raise ValueError(
                'Only the ShapeNet HDF5 data format is supported; '
                f'unsupported dataset_format={dataset_format!r}'
            )
        self.dataset_format = dataset_format
        self.preprocess_options = preprocess_options

        self.category_names = None
        if self.is_generation:
            self.category_names = ShapeNetCategoryNames(shapenet_metadata_path)

    @property
    def is_generation(self):
        return self.num_generation_samples > 0

    def get_generation_outdirs(self, batch):
        return 'none'

    def train_dataloader(self):
        dataset = self.get_dataset('train')
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          prefetch_factor=self.prefetch_factor,
                          drop_last=True,
                          shuffle=True,
                          num_workers=self.num_workers,
                          )

    def val_dataloader(self):
        return self.eval_dataloader(self.model_selection_split)

    def test_dataloader(self):
        return self.eval_dataloader(self.evaluation_split)

    def eval_dataloader(self, split: str):
        split = self._validate_eval_split(split, 'split')
        dataset = self.get_dataset(split)
        num_workers = min(self.eval_batch_size, self.num_workers)
        return DataLoader(dataset,
                          batch_size=self.eval_batch_size,
                          drop_last=False,
                          shuffle=False,
                          num_workers=num_workers,
                          )

    @staticmethod
    def _validate_eval_split(split, name):
        if split not in {'val', 'test'}:
            raise ValueError(f'{name} must be either val or test, got {split!r}')
        return split

    def get_dataset(self, split: str):
        if self.is_generation and (split != 'train'):
            return ShapeNetCategorySamplingDataset(categories=self.categories,
                                                   num_samples=self.num_generation_samples)

        if split == 'train':
            transform = AxisScaling(interval=(0.75, 1.25), jitter=True)
            repeat = self.repeat
            num_query_points = self.num_query_points
        else:
            transform = None
            repeat = 1
            num_query_points = -1

        return ShapeNetOccupancyDataset(
            root_dir=self.root_dir,
            split=split,
            transform=transform,
            repeat=repeat,
            num_query_points=num_query_points,
            categories=self.categories,
            use_queries=self.use_queries,
            use_full_surface=self.use_full_surface,
            pc_size=self.pc_size,
            chunk_size=self.chunk_size,
            oversample_ratio=self.oversample_ratio,
        )

    def available_splits(self):
        return ['train', 'val', 'test']

    def preprocess_data(self, split: str):
        os.makedirs(self.root_dir, exist_ok=True)
        query_root_dir = path.join(self.raw_root_dir, 'ShapeNetV2_point')
        surface_root_dir = path.join(self.raw_root_dir, 'ShapeNetV2_surface')

        ## identify items
        items = []
        class_dirs = [x for x in os.listdir(query_root_dir) if os.path.isdir(path.join(query_root_dir, x))]
        for class_dir in class_dirs:
            split_list_path = path.join(query_root_dir, class_dir, f'{split}.lst')
            with open(split_list_path, 'r') as f:
                lines = f.read().split('\n')

            items.extend([(class_dir, x.replace('.npz', '')) for x in lines if x != ''])
            pass

        file = h5py.File(path.join(self.root_dir, f'{split}.h5'), 'w')
        failed_count = 0
        for item in tqdm(items):
            class_id, object_id = item
            try:
                query_data = np.load(path.join(query_root_dir, class_id, f'{object_id}.npz'))
                surface_data = np.load(path.join(surface_root_dir, class_id, '4_pointcloud', f'{object_id}.npz'))
                scale_data = np.load(path.join(query_root_dir, class_id, f'{object_id}.npy'))
                vol_points, vol_label = query_data['vol_points'], query_data['vol_label']
                near_points, near_label = query_data['near_points'], query_data['near_label']
                surface_points = surface_data['points']
            except Exception as e:
                print(e)
                failed_count += 1
                continue

            group = file.create_group(f'{class_id}/{object_id}')
            query_dtype = np.float16 if split == 'train' else np.float32
            group['vol_points'] = vol_points.astype(query_dtype)
            group['vol_label'] = vol_label
            group['near_points'] = near_points.astype(query_dtype)
            group['near_label'] = near_label
            group['surface_points'] = surface_points.astype(np.float32)
            group.attrs['scale'] = scale_data.item()

        file.close()
        print(f'Preprocessing {split} done. Failed {failed_count} items.')


SHAPENET_CATEGORY_IDS = [
    '02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884',
    '02843684', '02871439', '02876657', '02880940', '02924116', '02933112', '02942699',
    '02946921', '02954340', '02958343', '02992529', '03001627', '03046257', '03085013',
    '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137',
    '03593526', '03624134', '03636649', '03642806', '03691459', '03710193', '03759954',
    '03761084', '03790512', '03797390', '03928116', '03938244', '03948459', '03991062',
    '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267',
    '04379243', '04401088', '04460130', '04468005', '04530566', '04554684'
]


class ShapeNetOccupancyDataset(HDF5Dataset):

    def __init__(self,
                 root_dir: str,
                 split: str = None,
                 categories: List[str] = None,
                 transform: Callable = None,
                 use_queries: bool = True,
                 use_full_surface: bool = False,
                 num_query_points: Union[int, Tuple[int, int]] = 1024,
                 pc_size: int = 2048,
                 repeat: int = 16,
                 chunk_size: int = 2000,
                 oversample_ratio: int = 10,
                 ):
        file_path = path.join(root_dir, f'{split}.h5')
        super().__init__(file_path)
        if isinstance(num_query_points, int):
            num_query_points = (num_query_points, num_query_points)

        self.pc_size = pc_size
        self.transform = transform
        self.num_volume_query_points, self.num_near_query_points = num_query_points
        self.split = split
        self.oversample_ratio = oversample_ratio
        self.chunk_size = chunk_size
        self.repeat = repeat
        self.use_queries = use_queries
        self.use_full_surface = use_full_surface

        if categories is None:
            categories = SHAPENET_CATEGORY_IDS
        categories.sort()

        self.items = []
        for category in categories:
            object_ids = self.file[category].keys()
            self.items.extend([(category, object_id) for object_id in object_ids])

        self.close_file()

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.items)
        category, object_id = self.items[idx]
        file = self.file
        group = file[f'{category}/{object_id}']

        vol_points = group['vol_points']
        vol_label = group['vol_label']
        near_points = group['near_points']
        near_label = group['near_label']
        scale = float(group.attrs['scale'])
        surface = group['surface_points']
        full_surface = surface[:]
        ind = np.random.choice(surface.shape[0], self.pc_size, replace=False)
        surface = full_surface[ind]

        surface = surface * scale
        if self.use_full_surface:
            full_surface = full_surface * scale
        surface = torch.from_numpy(surface.astype(np.float32))

        if self.use_queries:
            if self.num_volume_query_points > 0:
                vol_points, vol_label = two_stage_sampling([vol_points, vol_label],
                                                           num_samples=self.num_volume_query_points,
                                                           chunk_size=self.chunk_size,
                                                           oversample_ratio=self.oversample_ratio,
                                                           )
                near_points, near_label = two_stage_sampling([near_points, near_label],
                                                             num_samples=self.num_near_query_points,
                                                             chunk_size=self.chunk_size,
                                                             oversample_ratio=self.oversample_ratio,
                                                             )
            else:
                vol_points = vol_points[:]
                vol_label = vol_label[:]
                near_points = None
                near_label = None

            vol_points = torch.from_numpy(vol_points).float()
            vol_label = torch.from_numpy(vol_label).float()
            if near_points is not None:
                near_points = torch.from_numpy(near_points)
                near_label = torch.from_numpy(near_label).float()
                query_points = torch.cat([vol_points, near_points], dim=0)
                labels = torch.cat([vol_label, near_label], dim=0)
            else:
                query_points = vol_points
                labels = vol_label
        else:
            query_points = None
            labels = None

        max_val = torch.abs(surface).max().item()
        if self.transform is not None:
            surface, query_points, max_val = self.transform(surface, query_points)

        item = {
            'idx': idx,
            'surface': surface,
            'query_points': query_points,
            'labels': labels,
            'category_ids': SHAPENET_CATEGORY_IDS.index(category),
            'source': 'shapenet',
            'category': category,
            'object_id': object_id,
            'max_val': max_val,
            'num_vol_points': self.num_volume_query_points,
        }
        if self.use_full_surface:
            full_surface = torch.from_numpy(full_surface.copy().astype(np.float32))
            full_surface = full_surface.float()
            item['full_surface'] = full_surface

        return item


class ShapeNetCategoryNames:
    def __init__(self, metadata_path):
        super().__init__()

        with open(metadata_path, 'r') as f:
            data = json.load(f)
        self.data = data

    def get_name(self, idx: int):
        cat_id = SHAPENET_CATEGORY_IDS[idx]
        return self.data[cat_id]


class ShapeNetCategorySamplingDataset(Dataset):
    def __init__(self, categories: List[str] = None, num_samples: int = 1000):
        if categories is None:
            categories = SHAPENET_CATEGORY_IDS
        categories.sort()

        self.categories = categories
        self.num_samples = num_samples

    def __len__(self):
        return len(self.categories) * self.num_samples

    def __getitem__(self, idx: int):
        category_idx = idx // self.num_samples
        inner_idx = idx % self.num_samples
        category = self.categories[category_idx]
        return {
            'idx': idx,
            'inner_idx': inner_idx,
            'surface': torch.zeros(1, 3).float(),
            'category_ids': SHAPENET_CATEGORY_IDS.index(category),
        }
