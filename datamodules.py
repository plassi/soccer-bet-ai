from pytorch_lightning.core.datamodule import LightningDataModule
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
        # self.dataset_1 = ApiFootballDataset(dataframe=df, dataset='lstm')
        self.dataset_2 = ApiFootballDataset(dataframe=df)


    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        train_idx = [i for i in range(0, len(self.dataset_2) - 480)]
        val_idx = [i for i in range(len(self.dataset_2) - 480, len(self.dataset_2) - 240)]
        test_idx = [i for i in range(len(self.dataset_2) - 240, len(self.dataset_2))]

        # self.train_1 = Subset(self.dataset_1, train_idx)
        # self.val_1 = Subset(self.dataset_1, val_idx)
        # self.test_1 = Subset(self.dataset_1, test_idx)
        
        self.train_2 = Subset(self.dataset_2, train_idx)
        self.val_2 = Subset(self.dataset_2, val_idx)
        self.test_2 = Subset(self.dataset_2, test_idx)

    def train_dataloader(self):
        # dataloader_1 = DataLoader(self.train_1, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        dataloader_2 = DataLoader(self.train_2, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        # return [dataloader_1, dataloader_2]
        return dataloader_2

    def val_dataloader(self):
        # dataloader_1 = DataLoader(self.val_1, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        dataloader_2 = DataLoader(self.val_2, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        # return [dataloader_1, dataloader_2]
        return dataloader_2

    def test_dataloader(self):
        # dataloader_1 = DataLoader(self.test_1, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        dataloader_2 = DataLoader(self.test_2, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        # return [dataloader_1, dataloader_2]
        return dataloader_2