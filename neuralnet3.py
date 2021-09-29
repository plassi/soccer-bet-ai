import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


# %%
class FootballOddsDecoder(pl.LightningModule):
    def __init__(self, batch_size, learning_rate):
        super().__init__()

        self.lr = learning_rate
        self.batch_size = batch_size
        self.dropout = 0.2

        # Pool players data to 48 features
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(48),
        )

        self.ff = nn.Sequential(
            nn.Linear(in_features=173, out_features=128),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=3),
            nn.ReLU(),
        )

        

    def forward(self, X1, X2):

        X1_pooled = self.pool(X1)
        X = torch.cat((X1_pooled, X2), dim=2)

        return self.ff(X)

    def training_step(self, batch, batch_idx):

        X1, X2, y = batch[0], batch[1], batch[2]

        y_ff_hat = self.forward(X1, X2)

        loss_ff = F.mse_loss(y_ff_hat, y.float())

        # Logging to TensorBoard by default
        self.log("train/loss/ff", loss_ff)
        self.log("learning_rate", self.lr)
        self.log("batch_size", self.batch_size)
        self.log("dropout", self.dropout)

        return loss_ff


    # Create validation step
    def validation_step(self, batch, batch_idx):

        X1, X2, y = batch[0], batch[1], batch[2]

        y_ff_hat = self.forward(X1, X2)

        loss_ff = F.mse_loss(y_ff_hat, y.float())
        self.log("val/loss/ff", loss_ff)
        

    # # Create test step
    # def test_step(self, batch, batch_idx):

    #     X, y = batch[0], batch[1]

    #     y_ff_hat = self.ff(X)

    #     loss_ff = F.mse_loss(y_ff_hat, y.float())
    #     self.log("test/loss/ff", loss_ff)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer