from abc import ABC, abstractmethod

from lightning.pytorch import LightningDataModule


class BaseDataModule(LightningDataModule, ABC):
    batch_size: int
    eval_grid_size: int

    @abstractmethod
    def get_dataset(self, split: str):
        raise NotImplementedError

    @abstractmethod
    def eval_dataloader(self, split: str):
        raise NotImplementedError

    @abstractmethod
    def get_generation_outdirs(self, batch):
        raise NotImplementedError

    def available_splits(self):
        return []

    def preprocess_data(self, split: str):
        pass

    def preprocess(self):
        splits = self.available_splits()
        for split in splits:
            self.preprocess_data(split)
