import math
import pickle
from pytorch_lightning.core.datamodule import LightningDataModule
import torch
import random
from torch.utils.data import DataLoader, Subset
from dataset.api_football_dataset_ff import ApiFootballDataset
import os
from tqdm import tqdm


class FootballOddsDataModule(LightningDataModule):
    def __init__(self, batch_size, n_workers, random_seed=None):
        super().__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers

        if random_seed is not None:
            self.random_seed = random_seed
        else:
            self.random_seed = random.randint(0, 2**16)

    def prepare_data(self, datapath, feature_set, cache):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # if datapath contains a file that starts with 'dataset_', load that file
        # else, create dataset

        dataset_filename = 'dataset' + '_features_set_' + \
            str(feature_set) + '.pkl'

        if os.path.isfile(os.path.join(datapath, dataset_filename)) and cache == True:
            print("load dataset from cache")
            with open(datapath + dataset_filename, 'rb') as f:
                print(f)
                self.dataset = pickle.load(f)
        else:
            # load df from pickle
            files = [f for f in os.listdir(
                datapath) if f.startswith("fixtures")]
            files = [os.path.join(datapath, f) for f in files]

            df = pickle.load(open(files[0], 'rb'))
            for f in tqdm(files[1:]):
                df = df.append(pickle.load(open(f, 'rb')))

            # arrange df in chronological order by fixture_timestamp
            df = df.sort_values(by=['fixture_timestamp'])

            self.dataset = ApiFootballDataset(df=df, feature_set=feature_set)
            if cache == True:
                with open(datapath + dataset_filename, 'wb') as f:
                    pickle.dump(self.dataset, f)


        dataset_length = len(self.dataset)

        print("Dataset length: ", dataset_length)

        # Split dataset into train and val
        train_idx = [i for i in range(
            0, math.ceil(dataset_length * 0.8))]

        # print("Train idx: ", train_idx)
        val_idx = [i for i in range(
            math.ceil(dataset_length * 0.8), dataset_length)]

        self.train_set = Subset(self.dataset, train_idx)
        self.val_set = Subset(self.dataset, val_idx)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        pass

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        dataloader = DataLoader(self.train_set, batch_size=self.batch_size,
                                num_workers=self.n_workers, shuffle=True, generator=generator)
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
