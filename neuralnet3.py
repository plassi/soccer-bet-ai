import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


# %%
class FootballOddsDecoder(pl.LightningModule):
    def __init__(self, batch_size, learning_rate):
        super().__init__()

        # print("Initializing LSTM")

        self.lr = learning_rate
        self.batch_size = batch_size
        self.dropout = 0.2

        self.ff = nn.Sequential(
            nn.Linear(in_features=59325, out_features=1024),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=3),
            nn.ReLU(),
        )

        

    def forward(self, X):

        return self.ff(X)

    def training_step(self, batch, batch_idx):


        X, y = batch[0], batch[1]

        self.y_ff_hat = self.ff(X)

        loss_ff = F.mse_loss(self.y_ff_hat, y.float())

        # Logging to TensorBoard by default
        self.log("train_loss", loss_ff)

        return loss_ff

    # def training_epoch_end(self, outputs) -> None:
    #     print(self.y_ff_hat)

    # Create validation step
    def validation_step(self, batch, batch_idx):

        X, y = batch[0], batch[1]

        y_ff_hat = self.ff(X)

        loss_ff = F.mse_loss(y_ff_hat, y.float())
        
        return {"loss": loss_ff}

    def validation_epoch_end(self, outputs):
        
        self.log("hp_learning_rate", self.lr)
        self.log("hp_batch_size", self.batch_size)
        self.log("hp_dropout", self.dropout)
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        print("avg_loss:", avg_loss)

        return {'val_loss': avg_loss, 'log': {'val_loss': avg_loss}}
        

    # Create test step
    
    # def test_step(self, batch, batch_idx):

    #     X, y = batch[0], batch[1]

    #     y_ff_hat = self.ff(X)

    #     loss_ff = F.mse_loss(y_ff_hat, y.float())
    #     self.log("test/loss/ff", loss_ff)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer