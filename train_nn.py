# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

from argparse import ArgumentParser

from dataset.helpers.load_data import Load_data
from dataset.api_football_dataset import ApiFootballDataset


# %%


# %%
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule

from torchinfo import summary



# %%
class DataModule(LightningDataModule):
    def __init__(self, batch_size=32, n_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.n_workers = n_workers

    def prepare_data(self, datapath):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        df = Load_data(csv_data_path=datapath).get_data()
        self.dataset_1 = ApiFootballDataset(dataframe=df, Xy_pair='1',)
        self.dataset_2 = ApiFootballDataset(dataframe=df, Xy_pair='2',)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        train_idx = [i for i in range(0, len(self.dataset_1) - 600)]
        val_idx = [i for i in range(len(self.dataset_1) - 600, len(self.dataset_1) - 300)]
        test_idx = [i for i in range(len(self.dataset_1) - 300, len(self.dataset_1))]

        self.train_1 = Subset(self.dataset_1, train_idx)
        self.train_2 = Subset(self.dataset_2, train_idx)
        self.val_1 = Subset(self.dataset_1, val_idx)
        self.val_2 = Subset(self.dataset_2, val_idx)
        self.test_1 = Subset(self.dataset_1, test_idx)
        self.test_2 = Subset(self.dataset_2, test_idx)

    def train_dataloader(self):
        dataloader_1 = DataLoader(self.train_1, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        dataloader_2 = DataLoader(self.train_2, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        return [dataloader_1, dataloader_2]

    def val_dataloader(self):
        dataloader_1 = DataLoader(self.val_1, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        dataloader_2 = DataLoader(self.val_2, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        return [dataloader_1, dataloader_2]

    def test_dataloader(self):
        dataloader_1 = DataLoader(self.test_1, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        dataloader_2 = DataLoader(self.test_2, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=False)
        return [dataloader_1, dataloader_2]

    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass


# %%


# %%
def transform_tensor(x):
    x = x.view(x.size(0), -1)
    return x.float()


# %%
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, enc_in, enc_out, dec_in, dec_out):
        super().__init__()

        print("Initializing LitAutoEncoder")

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        enc_D_in, enc_H, enc_D_out = enc_in, 128, enc_out
        dec_D_in, dec_H, dec_D_out = dec_in, 16, dec_out

        self.encoder = nn.Sequential(
            nn.Linear(in_features=enc_D_in, out_features=enc_H),
            nn.ReLU(),
            nn.Linear(in_features=enc_H, out_features=enc_D_out)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=dec_D_in, out_features=dec_H),
            nn.ReLU(),
            nn.Linear(in_features=dec_H, out_features=dec_D_out)
        )

        print(summary(self.encoder).summary_list)
        print(summary(self.decoder).summary_list)

        

    def forward(self, x_1):
        # in lightning, forward defines the prediction/inference actions
        embedding_1 = self.encoder(x_1)
        embedding_2 = self.decoder(embedding_1)
        return embedding_2

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        X_1, y_1 = batch[0][0], batch[0][1]
        _, y_2 = batch[1][0], batch[1][1]

        # Transform tensors
        X_1 = transform_tensor(X_1)
        y_1 = transform_tensor(y_1)
        y_2 = transform_tensor(y_2)

        # Forward pass encoder
        y_1_hat = self.encoder(X_1)

        # Forward pass decoder
        y_2_hat = self.decoder(y_1_hat)

        # Loss
        loss_1 = F.mse_loss(y_1_hat, y_1)
        loss_2 = F.mse_loss(y_2_hat, y_2)

        # Logging to TensorBoard by default
        self.log("train_loss_1", loss_1)
        self.log("train_loss_2", loss_2)
        self.log("train_loss", loss_1 + loss_2)

        return loss_1 + loss_2

    # Create validation step
    def validation_step(self, batch, batch_idx, dataset_idx):
        # batch is a tuple of (X, y)
        if dataset_idx == 0:
            X, y = batch[0], batch[1]

            # Transform tensors
            X = transform_tensor(X)
            y = transform_tensor(y)

            # Forward pass encoder
            y_hat = self.encoder(X)

            # Loss
            loss = F.mse_loss(y_hat, y)

            # Logging to TensorBoard by default
            self.log("val_loss_1", loss)
            return loss
        if dataset_idx == 0:
            X, y = batch[0]

            # Forward pass encoder
            y_hat = self.decoder(X)

            # Loss
            loss = F.mse_loss(y_hat, y)

            # Logging to TensorBoard by default
            self.log("val_loss_2", loss)
            return loss

    # Create test step
    
    def test_step(self, batch, batch_idx, dataset_idx):
        if dataset_idx == 0:
            X, y = batch[0], batch[1]

            # Transform tensors
            X = transform_tensor(X)
            y = transform_tensor(y)

            # Forward pass encoder
            y_hat = self.encoder(X)

            # Loss
            loss = F.mse_loss(y_hat, y)

            # Logging to TensorBoard by default
            self.log("test_loss_1", loss)
            return loss
        if dataset_idx == 0:
            X, y = batch[0]

            # Forward pass encoder
            y_hat = self.decoder(X)

            # Loss
            loss = F.mse_loss(y_hat, y)

            # Logging to TensorBoard by default
            self.log("test_loss_2", loss)
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# %%
# add arguments 
parser = ArgumentParser()
parser.add_argument('--datapath', default='..data/', type=str)
args = parser.parse_args()

# Get parameters
print('Load data to get neural network features...')

datapath = args.datapath

datamodule = DataModule(n_workers=2)
datamodule.prepare_data(datapath=datapath)


enc_in_features = len(datamodule.dataset_1[0][0][0])
enc_out_features = len(datamodule.dataset_1[0][1][0])
dec_in_features = len(datamodule.dataset_2[0][0][0])
dec_out_features = len(datamodule.dataset_2[0][1][0])


# init model
coder = LitAutoEncoder(enc_in=enc_in_features, enc_out=enc_out_features,
                       dec_in=dec_in_features, dec_out=dec_out_features, )

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs)
# trainer = Trainer(tpu_cores=1) (if you have TPUs)
trainer = pl.Trainer(min_epochs=3, max_epochs=30, progress_bar_refresh_rate=1, gpus=1, precision=16)
# precision=16 for GPU/TPU
trainer.fit(coder, datamodule)
