from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader

from datasets import V2XTrajDataset


class V2XTrajDataModule(LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 32,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        super(V2XTrajDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.local_radius = local_radius

    def prepare_data(self) -> None:
        V2XTrajDataset(self.root, 'val', self.val_transform, self.local_radius)
        V2XTrajDataset(self.root, 'train', self.train_transform, self.local_radius)
        #V2XTrajDataset(self.root, 'test', self.train_transform, self.local_radius)

    def setup(self, stage: Optional[str] = None) -> None:
        self.val_dataset = V2XTrajDataset(self.root, 'val', self.val_transform, self.local_radius)
        self.train_dataset = V2XTrajDataset(self.root, 'train', self.train_transform, self.local_radius)
        #self.test_dataset = V2XTrajDataset(self.root, 'test', self.test_transform, self.local_radius)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
