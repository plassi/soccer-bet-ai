import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


# %%
class FootballOddsDecoder(pl.LightningModule):
    def __init__(self, h_layers, h_features, batch_size, learning_rate, dropout):
        super().__init__()
        self.save_hyperparameters()
        # print("Initializing LSTM")

        self.lr = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout

        if h_layers == 0:
            self.ff = nn.Sequential(
                nn.Linear(in_features=12540, out_features=h_features),
                nn.Dropout(self.dropout),
                nn.ReLU(),

                nn.Linear(in_features=h_features, out_features=3),
                nn.Softmax(dim=2)
            )

        if h_layers == 1:
            self.ff = nn.Sequential(
                nn.Linear(in_features=79135, out_features=h_features),
                nn.Dropout(self.dropout),
                nn.ReLU(),

                nn.Linear(in_features=h_features, out_features=3),
                nn.Softmax(dim=2)
            )

        if h_layers == 2:
            self.ff = nn.Sequential(
                nn.Linear(in_features=79135, out_features=h_features),
                nn.Dropout(self.dropout),
                nn.ReLU(),

                nn.Linear(in_features=h_features, out_features=h_features),
                nn.Dropout(self.dropout),
                nn.ReLU(),

                nn.Linear(in_features=h_features, out_features=3),
                nn.Softmax(dim=2)
            )

    def forward(self, X):

        return self.ff(X)

    def training_step(self, batch, batch_idx):

        X, y = batch[0], batch[1]

        y_ff_hat = self.ff(X)

        loss_ff = F.mse_loss(y_ff_hat, y)

        # Logging to TensorBoard by default
        self.log("train_loss", loss_ff)

        return loss_ff

    # def training_epoch_end(self, outputs) -> None:
    #     print(self.y_ff_hat)

    # Create validation step
    def validation_step(self, batch, batch_idx):

        X, y = batch[0], batch[1]

        y_ff_hat = self.ff(X)

        loss_ff = F.mse_loss(y_ff_hat, y)

        y_ff_hat_argmax = torch.argmax(y_ff_hat, dim=2)

        hits = 0
        misses = 0

        for i, y_s in enumerate(y):
            if y_s[0][y_ff_hat_argmax[i]] == 0:
                misses += 1
            else:
                hits += 1

        return {"loss": loss_ff, "y_hat": y_ff_hat, "hits": hits, "misses": misses}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # count mean and deviations from the mean for every column in each "y" in output
        y_ff_hat = torch.stack([x['y_hat'] for x in outputs]).mean(dim=0)
        y_ff_std = torch.stack([x['y_hat'] for x in outputs]).std(dim=0)

        # count single mean and deviation for every column in "y" in output
        y_ff_hat_mean = y_ff_hat.mean(dim=0)
        y_ff_std_mean = y_ff_std.mean(dim=0)

        hits = sum([x['hits'] for x in outputs])
        misses = sum([x['misses'] for x in outputs])
        percentage = hits / (hits + misses)

        # Print and log values

        print(f"\n1 mean: {y_ff_hat_mean[0][0]}")
        print(f"X mean: {y_ff_hat_mean[0][1]}")
        print(f"2 mean: {y_ff_hat_mean[0][2]}")

        print(f"1 std: {y_ff_std_mean[0][0]}")
        print(f"X std: {y_ff_std_mean[0][1]}")
        print(f"2 std: {y_ff_std_mean[0][2]}")

        print(f"\nval_loss: {avg_loss.item()}")
        print(f"val_accuracy: {percentage:2f}")
        mean_of_std_means = y_ff_std_mean.mean()
        print(f"mean_of_std_means: {mean_of_std_means}")

        # self.log("val_means", {
        #          "val_1_mean": y_ff_hat_mean[0][0],
        #          "val_X_mean": y_ff_hat_mean[0][1],
        #          "val_2_mean": y_ff_hat_mean[0][2]})

        self.log("mean_of_std_means", mean_of_std_means)

        # self.log("val_stds", {
        #     "val_1_std": y_ff_std_mean[0][0],
        #     "val_X_std": y_ff_std_mean[0][1],
        #     "val_2_std": y_ff_std_mean[0][2],
        #     })

        self.log("val_1_mean", y_ff_hat_mean[0][0])
        self.log("val_X_mean", y_ff_hat_mean[0][1])
        self.log("val_2_mean", y_ff_hat_mean[0][2])

        self.log("val_1_std", y_ff_std_mean[0][0])
        self.log("val_X_std", y_ff_std_mean[0][1])
        self.log("val_2_std", y_ff_std_mean[0][2])

        self.log("val_loss", avg_loss)
        self.log("val_accuracy", percentage)

        return avg_loss

    # Create test step

    def test_step(self, batch, batch_idx):
        X, y = batch[0], batch[1]

        y_ff_hat = self.ff(X)
        y_ff_hat = torch.argmax(y_ff_hat, dim=2)

        hits = 0
        misses = 0

        for i, y_s in enumerate(y):
            if y_s[0][y_ff_hat[i]] == 0:
                misses += 1
            else:
                hits += 1

        # print(f"Hits: {hits}")
        # print(f"Misses: {misses}")

        return {"hits": hits, "misses": misses}

    def test_epoch_end(self, outputs):
        hits = sum([x['hits'] for x in outputs])
        misses = sum([x['misses'] for x in outputs])
        percentage = hits / (hits + misses)
        print(f"\nHits: {hits}")
        print(f"\nMisses: {misses}")
        print(f"\nTest accuracy: {percentage:2f}")

        return percentage

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
