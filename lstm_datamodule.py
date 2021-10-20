import math
import pickle
from pytorch_lightning.core.datamodule import LightningDataModule
import torch
import random
from torch.utils.data import DataLoader, Subset
from dataset.api_football_dataset3 import ApiFootballDataset


class FootballOddsDataModule(LightningDataModule):
    def __init__(self, batch_size, n_workers, random_seed=None):
        super().__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers

        if random_seed is not None:
            self.random_seed = random_seed
        else:
            self.random_seed = random.randint(0, 2**16)

    def prepare_data(self, datapath):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # load df from pickle
        df = pickle.load(open(datapath + "/df.pickle", 'rb'))
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

        self.train_set = Subset(self.dataset, train_idx)
        self.val_set = Subset(self.dataset, val_idx)
        # self.test = Subset(self.dataset, test_idx)

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        dataloader = DataLoader(self.train_set, batch_size=self.batch_size,
                                num_workers=self.n_workers, shuffle=False, generator=generator)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False, drop_last=True)
        return dataloader

    def test_dataloader(self):
        # dataloader_1 = DataLoader(self.test_1, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        dataloader = DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        return dataloader
