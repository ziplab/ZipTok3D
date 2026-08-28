import csv
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .base import BaseDataModule
from .transform import AxisScaling
from .utils import two_stage_sampling


class TrellisOccupancyDataModule(BaseDataModule):
    """Data module for the sharded output of tools/trellis500k_preprocess.py."""

    def __init__(
        self,
        root_dir: str,
        num_workers: int = 4,
        batch_size: int = 16,
        eval_batch_size: int = -1,
        prefetch_factor: int = 2,
        num_query_points: int = 4096,
        repeat: int = 1,
        pc_size: int = 2048,
        chunk_size: int = 5000,
        oversample_ratio: int = 3,
        use_queries: bool = True,
        use_full_surface: bool = False,
        max_open_shards: int = 2,
        model_selection_split: str = "val",
        evaluation_split: str = "test",
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.eval_grid_size = 128
        self.num_workers = int(num_workers)
        self.batch_size = int(batch_size)
        self.eval_batch_size = (
            int(eval_batch_size) if eval_batch_size > 0 else self.batch_size
        )
        self.prefetch_factor = int(prefetch_factor)
        self.num_query_points = int(num_query_points)
        self.repeat = int(repeat)
        self.pc_size = int(pc_size)
        self.chunk_size = int(chunk_size)
        self.oversample_ratio = int(oversample_ratio)
        self.use_queries = bool(use_queries)
        self.use_full_surface = bool(use_full_surface)
        self.max_open_shards = int(max_open_shards)
        self.model_selection_split = self._validate_eval_split(
            model_selection_split, "model_selection_split"
        )
        self.evaluation_split = self._validate_eval_split(
            evaluation_split, "evaluation_split"
        )

    def get_generation_outdirs(self, batch):
        return "none"

    def _loader(self, split, *, training):
        dataset = self.get_dataset(split)
        workers = self.num_workers if training else min(self.eval_batch_size, self.num_workers)
        kwargs = {}
        if workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            dataset,
            batch_size=self.batch_size if training else self.eval_batch_size,
            drop_last=training,
            shuffle=training,
            num_workers=workers,
            **kwargs,
        )

    def train_dataloader(self):
        return self._loader("train", training=True)

    def val_dataloader(self):
        return self.eval_dataloader(self.model_selection_split)

    def test_dataloader(self):
        return self.eval_dataloader(self.evaluation_split)

    def eval_dataloader(self, split: str):
        split = self._validate_eval_split(split, "split")
        return self._loader(split, training=False)

    @staticmethod
    def _validate_eval_split(split, name):
        if split not in {"val", "test"}:
            raise ValueError(f"{name} must be either val or test, got {split!r}")
        return split

    def get_dataset(self, split: str):
        if split == "train":
            transform = AxisScaling(interval=(0.75, 1.25), jitter=True)
            repeat = self.repeat
            num_query_points = self.num_query_points
        else:
            transform = None
            repeat = 1
            num_query_points = -1
        return TrellisOccupancyDataset(
            root_dir=self.root_dir,
            split=split,
            transform=transform,
            repeat=repeat,
            num_query_points=num_query_points,
            pc_size=self.pc_size,
            chunk_size=self.chunk_size,
            oversample_ratio=self.oversample_ratio,
            use_queries=self.use_queries,
            use_full_surface=self.use_full_surface,
            max_open_shards=self.max_open_shards,
        )

    def available_splits(self):
        return ["train", "val", "test"]

    def preprocess_data(self, split: str):
        index = self.root_dir / f"{split}.csv"
        if not index.is_file():
            raise FileNotFoundError(
                f"missing {index}; run tools/trellis500k_preprocess.py process, "
                "split, and pack first"
            )


class TrellisOccupancyDataset(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str,
        transform: Callable = None,
        use_queries: bool = True,
        use_full_surface: bool = False,
        num_query_points: Union[int, Tuple[int, int]] = 4096,
        pc_size: int = 2048,
        repeat: int = 1,
        chunk_size: int = 5000,
        oversample_ratio: int = 3,
        max_open_shards: int = 2,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_queries = bool(use_queries)
        self.use_full_surface = bool(use_full_surface)
        self.pc_size = int(pc_size)
        self.repeat = int(repeat)
        self.chunk_size = int(chunk_size)
        self.oversample_ratio = int(oversample_ratio)
        self.max_open_shards = max(1, int(max_open_shards))
        if isinstance(num_query_points, int):
            num_query_points = (num_query_points, num_query_points)
        self.num_volume_query_points, self.num_near_query_points = num_query_points

        index_path = self.root_dir / f"{split}.csv"
        with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
            self.items = list(csv.DictReader(stream))
        self._files = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = OrderedDict()
        return state

    def __del__(self):
        for handle in getattr(self, "_files", {}).values():
            handle.close()

    def _file(self, shard: str):
        path = str((self.root_dir / shard).resolve())
        handle = self._files.pop(path, None)
        if handle is None:
            handle = h5py.File(path, "r", swmr=True)
        self._files[path] = handle
        while len(self._files) > self.max_open_shards:
            _, old = self._files.popitem(last=False)
            old.close()
        return handle

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, index):
        item_index = index % len(self.items)
        item = self.items[item_index]
        group = self._file(item["shard"])[item["group"]]

        full_surface = group["surface_points"][:].astype(np.float32)
        replace = len(full_surface) < self.pc_size
        surface_indices = np.random.choice(len(full_surface), self.pc_size, replace=replace)
        surface = torch.from_numpy(full_surface[surface_indices])

        query_points = labels = None
        if self.use_queries:
            vol_points = group["vol_points"]
            vol_label = group["vol_label"]
            near_points = group["near_points"]
            near_label = group["near_label"]
            if self.num_volume_query_points > 0:
                vol_points, vol_label = two_stage_sampling(
                    [vol_points, vol_label],
                    num_samples=self.num_volume_query_points,
                    chunk_size=self.chunk_size,
                    oversample_ratio=self.oversample_ratio,
                )
                near_points, near_label = two_stage_sampling(
                    [near_points, near_label],
                    num_samples=self.num_near_query_points,
                    chunk_size=self.chunk_size,
                    oversample_ratio=self.oversample_ratio,
                )
                query_points = torch.cat(
                    [torch.from_numpy(vol_points).float(), torch.from_numpy(near_points).float()],
                    dim=0,
                )
                labels = torch.cat(
                    [torch.from_numpy(vol_label).float(), torch.from_numpy(near_label).float()],
                    dim=0,
                )
                num_vol_points = self.num_volume_query_points
            else:
                query_points = torch.from_numpy(vol_points[:]).float()
                labels = torch.from_numpy(vol_label[:]).float()
                num_vol_points = len(vol_points)
        else:
            num_vol_points = 0

        max_val = torch.abs(surface).max().item()
        if self.transform is not None:
            surface, query_points, max_val = self.transform(surface, query_points)

        result = {
            "idx": item_index,
            "surface": surface,
            "query_points": query_points,
            "labels": labels,
            "category_ids": 0,
            "max_val": max_val,
            "num_vol_points": num_vol_points,
            "source": item["source"],
            "object_id": item["object_id"],
        }
        if self.use_full_surface:
            result["full_surface"] = torch.from_numpy(full_surface)
        return result
