import math

from pytorch_lightning.core.datamodule import LightningDataModule
import torch
from torch.utils.data import DataLoader, Subset
from dataset.helpers.load_data import Load_data
from dataset.api_football_dataset3 import ApiFootballDataset


class FootballOddsDataModule(LightningDataModule):
    def __init__(self, batch_size, n_workers):
        super().__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers

    def prepare_data(self, datapath):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        df = Load_data(csv_data_path=datapath).get_data()
        self.dataset = ApiFootballDataset(df=df)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        train_idx = [i for i in range(
            0, math.ceil(len(self.dataset) * 0.8))]
        val_idx = [i for i in range(
            math.ceil(len(self.dataset) * 0.8), len(self.dataset))]
        # test_idx = [i for i in range(
        #     len(self.dataset) - 240, len(self.dataset))]

        self.train = Subset(self.dataset, train_idx)
        self.val = Subset(self.dataset, val_idx)
        # self.test = Subset(self.dataset, test_idx)

    def train_dataloader(self):
        generator = torch.Generator()
        # if self.random_seed is not None:
        #     generator.manual_seed(self.random_seed)
        dataloader = DataLoader(self.train, batch_size=self.batch_size,
                                num_workers=self.n_workers, shuffle=True, generator=generator)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False, drop_last=True)
        return dataloader

    def test_dataloader(self):
        # dataloader_1 = DataLoader(self.test_1, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        dataloader = DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        return dataloader
