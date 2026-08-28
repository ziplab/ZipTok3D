from os import path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .base import BaseDataModule


class ShapeNetLatentDataset(Dataset):
    def __init__(self, cache_dir, split, repeat=1):
        split_dir = path.join(cache_dir, split)
        self.latents = np.load(path.join(split_dir, "latents.npy"), mmap_mode="r")
        self.labels = np.load(path.join(split_dir, "labels.npy"), mmap_mode="r")
        if len(self.latents) != len(self.labels):
            raise ValueError(f"mismatched latent cache in {split_dir}")
        self.repeat = max(1, int(repeat))

    def __len__(self):
        return len(self.latents) * self.repeat

    def __getitem__(self, index):
        index %= len(self.latents)
        return {
            "latents": torch.from_numpy(np.array(self.latents[index], copy=True)).float(),
            "category_ids": torch.as_tensor(int(self.labels[index]), dtype=torch.long),
        }


class ShapeNetLatentDataModule(BaseDataModule):
    def __init__(
        self,
        cache_dir,
        batch_size=32,
        eval_batch_size=64,
        num_workers=8,
        prefetch_factor=2,
        train_repeat=1,
        pin_memory=True,
        persistent_workers=True,
        model_selection_split="test",
        evaluation_split="val",
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_repeat = train_repeat
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.model_selection_split = self._validate_eval_split(
            model_selection_split, "model_selection_split"
        )
        self.evaluation_split = self._validate_eval_split(
            evaluation_split, "evaluation_split"
        )

    def get_dataset(self, split):
        repeat = self.train_repeat if split == "train" else 1
        return ShapeNetLatentDataset(self.cache_dir, split, repeat=repeat)

    def _loader(self, split, training):
        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        return DataLoader(
            self.get_dataset(split),
            batch_size=self.batch_size if training else self.eval_batch_size,
            shuffle=training,
            drop_last=training,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **kwargs,
        )

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self.eval_dataloader(self.model_selection_split)

    def test_dataloader(self):
        return self.eval_dataloader(self.evaluation_split)

    def eval_dataloader(self, split: str):
        split = self._validate_eval_split(split, "split")
        return self._loader(split, False)

    @staticmethod
    def _validate_eval_split(split, name):
        if split not in {"val", "test"}:
            raise ValueError(f"{name} must be either val or test, got {split!r}")
        return split

    def available_splits(self):
        return ["train", "val", "test"]

    def get_generation_outdirs(self, batch):
        return "none"
