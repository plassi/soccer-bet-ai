import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


# %%
class FootballOddsDecoder(pl.LightningModule):
    def __init__(self, batch_size, learning_rate, dropout):
        super().__init__()

        # print("Initializing LSTM")

        self.lr = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout

        self.ff = nn.Sequential(
            nn.Linear(in_features=79135, out_features=1024),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=3),
            nn.Sigmoid(),
        )

        

    def forward(self, X):

        return self.ff(X)

    def training_step(self, batch, batch_idx):


        X, y = batch[0], batch[1]

        y_ff_hat = self.ff(X)

        loss_ff = F.binary_cross_entropy(y_ff_hat, y.float())

        # Logging to TensorBoard by default
        self.log("train_loss", loss_ff)

        return loss_ff

    # def training_epoch_end(self, outputs) -> None:
    #     print(self.y_ff_hat)

    # Create validation step
    def validation_step(self, batch, batch_idx):

        X, y = batch[0], batch[1]

        y_ff_hat = self.ff(X)

        loss_ff = F.binary_cross_entropy(y_ff_hat, y.float())
        return {"loss": loss_ff, "y": y_ff_hat}

    def validation_epoch_end(self, outputs):
        
        self.log("hp_learning_rate", self.lr)
        self.log("hp_batch_size", self.batch_size)
        self.log("hp_dropout", self.dropout)
        
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # count mean and deviations from the mean for every column in each "y" in output
        y_ff_hat = torch.stack([x['y'] for x in outputs]).mean(dim=0)
        y_ff_std = torch.stack([x['y'] for x in outputs]).std(dim=0)

        # count single mean and deviation for every column in "y" in output
        y_ff_hat_mean = y_ff_hat.mean(dim=0)
        y_ff_std_mean = y_ff_std.mean(dim=0)

        # Print and log values

        print(f"\n1 mean: {y_ff_hat_mean[0][0]}")
        print(f"X mean: {y_ff_hat_mean[0][1]}")
        print(f"2 mean: {y_ff_hat_mean[0][2]}")

        print(f"1 std: {y_ff_std_mean[0][0]}")
        print(f"X std: {y_ff_std_mean[0][1]}")
        print(f"2 std: {y_ff_std_mean[0][2]}")

        print(f"\nval_loss: {avg_loss.item()}\n")
        
        self.log("val_1_mean", y_ff_hat_mean[0][0])
        self.log("val_X_mean", y_ff_hat_mean[0][1])
        self.log("val_2_mean", y_ff_hat_mean[0][2])

        self.log("val_1_std", y_ff_std_mean[0][0])
        self.log("val_X_std", y_ff_std_mean[0][1])
        self.log("val_2_std", y_ff_std_mean[0][2])

        self.log("val_loss", avg_loss)


        return {'val_loss': avg_loss, 'log': {'val_loss': avg_loss}}
        

    # Create test step
    
    def test_step(self, batch, batch_idx):
        X, y = batch[0], batch[1]


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer