import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


# class extractlastcell(nn.Module):
#     def forward(self, x):
#         out, _ = x
#         return out[:, -1, :]


# %%
class FootballOddsLSTM(pl.LightningModule):
    def __init__(self, h_layers, h_features, batch_size, learning_rate, dropout, input_features):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        # print("Initializing LSTM")

        if torch.cuda.is_available():
            self.lstm_hidden = (torch.randn(h_layers, 4, h_features, requires_grad=True, device='cuda'), torch.randn(
            h_layers, 4, h_features, requires_grad=True, device='cuda'))
        else:
            torch.device('cpu')

        self.lstm_hidden = (torch.randn(h_layers, 4, h_features, requires_grad=True, device='cpu'), torch.randn(
            h_layers, 4, h_features, requires_grad=True, device='cpu'))

        self.lr = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.input_size = input_features

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=h_features,
                            num_layers=h_layers, dropout=dropout, )

        self.ff = nn.Sequential(
            nn.Linear(in_features=h_features,
                      out_features=3),
            nn.Softmax(dim=1)
        )

    def forward(self, X):

        y_pred, self.lstm_hidden = self.lstm(X)
        y_ff_hat = self.ff(y_pred[:, -1, :])

        return y_ff_hat

    def training_step(self, batch, batch_idx):

        X, y = batch[0], batch[1]

        y_pred, _ = self.lstm(X, (self.lstm_hidden[0], self.lstm_hidden[1]))

        # print(self.lstm_hidden[1])

        y_ff_hat = self.ff(y_pred[:, -1, :])

        # print(y_ff_hat)

        opt = self.optimizers()
        opt.zero_grad()
        loss = F.mse_loss(y_ff_hat, y[:, :, -3:].view(-1, 3))
        self.manual_backward(loss, retain_graph=True)
        opt.step()

        # Logging to TensorBoard by default
        self.log("train_loss", loss)

        return loss

    # def training_epoch_end(self, outputs) -> None:
    #     print(self.y_ff_hat)

    # Create validation step
    def validation_step(self, batch, batch_idx):

        X, y = batch[0], batch[1]
        # print number of X features

        # print(X)
        # print("X.shape", X.shape)

        # print("lstm_hidden", self.lstm_hidden)
        # print("lstm_hidden.shape", self.lstm_hidden.shape)
        # print(X.size(-1))
        


        y_pred, self.lstm_hidden = self.lstm(X, (self.lstm_hidden[0], self.lstm_hidden[1]))

        y_ff_hat = self.ff(y_pred[:, -1, :])

        # print(y_ff_hat)

        loss_ff = F.mse_loss(y_ff_hat, y[:, :, -3:].view(-1, 3))

        # print("y_ff_hat", y_ff_hat)

        y_ff_hat_argmax = torch.argmax(y_ff_hat, dim=1)

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

        print(f"\n1 mean: {y_ff_hat_mean[0]}")
        print(f"X mean: {y_ff_hat_mean[1]}")
        print(f"2 mean: {y_ff_hat_mean[2]}")

        print(f"1 std: {y_ff_std_mean[0]}")
        print(f"X std: {y_ff_std_mean[1]}")
        print(f"2 std: {y_ff_std_mean[2]}")

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

        self.log("val_1_mean", y_ff_hat_mean[0])
        self.log("val_X_mean", y_ff_hat_mean[1])
        self.log("val_2_mean", y_ff_hat_mean[2])

        self.log("val_1_std", y_ff_std_mean[0])
        self.log("val_X_std", y_ff_std_mean[1])
        self.log("val_2_std", y_ff_std_mean[2])

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
        parameters = self.parameters()
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        return optimizer
        
