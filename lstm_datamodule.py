import math
import pickle
from pytorch_lightning.core.datamodule import LightningDataModule
import torch
import random
from torch.utils.data import DataLoader, Subset
from dataset.api_football_dataset_lstm import ApiFootballDataset

from tqdm import tqdm

import os


class FootballOddsDataModule(LightningDataModule):
    def __init__(self, batch_size, n_workers, random_seed=None, seq_length=4):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_workers = n_workers

        if random_seed is not None:
            self.random_seed = random_seed
        else:
            self.random_seed = random.randint(0, 2**16)

    def prepare_data(self, datapath):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # load files that start with "fixtures" from datapath
        files = [f for f in os.listdir(datapath) if f.startswith("fixtures")]
        files = [os.path.join(datapath, f) for f in files]

        df = pickle.load(open(files[0], 'rb'))
        for f in tqdm(files[1:]):
            df = df.append(pickle.load(open(f, 'rb')))

        # df = pickle.load(open(datapath + "/df.pickle", 'rb'))
        self.dataset = ApiFootballDataset(df=df)

        dataset_length = len(self.dataset)

        print("Dataset length: ", dataset_length)

        train_idx = [i for i in range(
            0, math.ceil(dataset_length * 0.8))]

        # print("Train idx: ", train_idx)
        val_idx = [i for i in range(
            math.ceil(dataset_length * 0.8), dataset_length)]

        self.train_set = Subset(self.dataset, train_idx)
        self.val_set = Subset(self.dataset, val_idx)
        # print("Val idx: ", val_idx)
        # test_idx = [i for i in range(
        #     len(self.dataset) - 240, len(self.dataset))]

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

        # self.test = Subset(self.dataset, test_idx)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_set, batch_size=self.batch_size,
                                num_workers=self.n_workers, shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False, drop_last=True)
        return dataloader

    def test_dataloader(self):
        # dataloader_1 = DataLoader(self.test_1, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        dataloader = DataLoader(
            self.val_set, batch_size=1, num_workers=self.n_workers, shuffle=False)
        return dataloader
