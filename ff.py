import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import pickle
import numpy as np

import matplotlib.pyplot as plt
from simulation.simulation import Simulation

# %%


class FootballOddsFF(pl.LightningModule):

    def __init__(self, h_layers, h_features, batch_size, learning_rate, dropout, input_features, datapath, feature_set):

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(
                    m.weight)
                torch.nn.init.zeros_(m.bias)

        super().__init__()
        # self.automatic_optimization = False
        self.save_hyperparameters()

        self.plot_every = 1

        self.h_layers = h_layers
        self.h_features = h_features
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.input_features = input_features
        self.datapath = datapath
        self.feature_set = feature_set

        self.simulation = Simulation()

        # Get simulation_data with pickle
        with open(str(datapath) + "simulation_dictionary_ff_" + str(feature_set) + ".pickle", "rb") as f:
            self.simulation_data = pickle.load(f)

        self.simulation_dictionaries = []
        for data in self.simulation_data:
            self.simulation_dictionaries.append(data[0])

        #

        # Create first layer
        self.first_layer = nn.Sequential(
            nn.Linear(in_features=self.input_features,
                      out_features=math.ceil(self.input_features / 4)),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )
        self.batch_first_layer = nn.BatchNorm1d(
            math.ceil(self.input_features / 4))

        self.second_layer = nn.Sequential(
            nn.Linear(in_features=math.ceil(self.input_features / 4),
                      out_features=self.h_features),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )
        self.batch_second_layer = nn.BatchNorm1d(
            self.h_features)

        # Create layers list
        self.hidden_layers = nn.ModuleList()

        # Create hidden layers
        for i in range(self.h_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(in_features=self.h_features,
                          out_features=self.h_features),
                nn.SiLU(),
                nn.Dropout(self.dropout),
            ))

        # Create batch normalization layers
        self.batch_layers = nn.ModuleList()
        for i in range(self.h_layers):
            self.batch_layers.append(nn.BatchNorm1d(self.h_features))

        # Create end layers
        self.second_last_layer = nn.Sequential(
            nn.Linear(in_features=math.ceil(self.h_features),
                      out_features=math.ceil(self.h_features / 4)),
            nn.SiLU(),
        )
        self.batch_second_last_layer = nn.BatchNorm1d(
            math.ceil(self.h_features / 4))

        self.last_layer = nn.Sequential(
            nn.Linear(in_features=math.ceil(self.h_features / 4),
                      out_features=3),
        )

        # initialize weights
        self.first_layer.apply(weights_init)
        self.second_layer.apply(weights_init)

        for i in range(self.h_layers):
            self.hidden_layers[i].apply(weights_init)

        self.second_last_layer.apply(weights_init)
        self.last_layer.apply(weights_init)

    def forward(self, X):

        y = self.first_layer(X)
        y = y.view(-1, math.ceil(self.input_features / 4))
        y = self.batch_first_layer(y)

        y = self.second_layer(y)
        y = y.view(-1, self.h_features)
        y = self.batch_second_layer(y)

        for i in range(self.h_layers):
            y = self.hidden_layers[i](y)
            y = y.view(-1, self.h_features)
            y = self.batch_layers[i](y)

        y = self.second_last_layer(y)
        y = y.view(-1, math.ceil(self.h_features / 4))
        y = self.batch_second_last_layer(y)

        y = self.last_layer(y)

        return y

    def training_step(self, batch, batch_idx):

        X, y = batch[0], batch[1]

        _, y = torch.max(y, dim=1)
        yx = y.clone().detach().type(torch.long)

        y_ff_hat = self.forward(X)
        # y_ff_hat = y_ff_hat.reshape(y_ff_hat.shape[0], y_ff_hat.shape[2])

        loss = F.cross_entropy(y_ff_hat, yx)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)

        return loss

    def custom_histogram_adder(self):

        # iterating through all parameters
        for name, params in self.named_parameters():
            # if params contains nan
            if not torch.isnan(params).any():
                self.logger.experiment.add_histogram(
                    name, params, self.current_epoch)

    def training_epoch_end(self, outputs):
        if self.current_epoch == 0:
            sampleImg = torch.rand((self.batch_size, self.input_features))
            self.logger.experiment.add_graph(FootballOddsFF(
                self.h_layers, self.h_features, self.batch_size, self.learning_rate, self.dropout, self.input_features, self.datapath, self.feature_set), sampleImg)

        # logging histograms
        self.custom_histogram_adder()

    # Create validation step

    def validation_step(self, batch, batch_idx):

        X, y = batch[0], batch[1]
        # print number of X features

        # print(X)
        # print("X.shape", X.shape)

        # print("y", y)
        # print("y.shape", y.shape)

        _, y = torch.max(y, dim=1)
        yx = y.clone().detach().type(torch.long)

        # print("yx", yx)
        # print("yx.shape", yx.shape)

        y_ff_hat = self.forward(X)
        # print("y_ff_hat", y_ff_hat)
        # print("y_ff_hat.shape", y_ff_hat.shape)

        # y_ff_hat = y_ff_hat.reshape(y_ff_hat.shape[0], y_ff_hat.shape[2])

        loss = F.cross_entropy(y_ff_hat, yx)

        # print(loss)

        y_hat = F.softmax(y_ff_hat, dim=-1)

        # print("y_ff_hat", y_ff_hat)
        # print("y_ff_hat.shape", y_ff_hat.shape)

        # Logging to TensorBoard by default
        self.log("val_loss", loss)

        # return loss

        return {"loss": loss, "y_hat": y_hat}

    def validation_epoch_end(self, outputs):

        # print("outputs", outputs)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # count mean and deviations from the mean for every column in each "y" in output
        # for x in outputs:
        #     print(x['y_hat'])

        y_ff_hat = torch.stack([x['y_hat'] for x in outputs]).mean(dim=0)
        y_ff_std = torch.stack([x['y_hat'] for x in outputs]).std(dim=0)

        # print("y_ff_hat.shape", y_ff_hat.shape)
        # # print("y_ff_hat", y_ff_hat)
        # print("y_ff_std.shape", y_ff_std.shape)
        # # print("y_ff_std", y_ff_std)

        # count single mean and deviation for every column in "y" in output

        y_ff_hat_mean = y_ff_hat.mean(dim=0)
        y_ff_std_mean = y_ff_std.mean(dim=0)
        # print("y_ff_hat_mean.shape", y_ff_hat_mean.shape)
        # print("y_ff_hat_mean", y_ff_hat_mean)
        # print("y_ff_std_mean.shape", y_ff_std_mean.shape)
        # print("y_ff_std_mean", y_ff_std_mean)

        # calculate mean of mean of stds
        mean_of_std_means = y_ff_std_mean.mean()
        # Print and log values

        print(f"\n1 mean: {y_ff_hat_mean[0]}")
        print(f"X mean: {y_ff_hat_mean[1]}")
        print(f"2 mean: {y_ff_hat_mean[2]}")

        print(f"1 std: {y_ff_std_mean[0]}")
        print(f"X std: {y_ff_std_mean[1]}")
        print(f"2 std: {y_ff_std_mean[2]}")

        print(f"\nval_loss: {avg_loss.item()}")
        print(f"mean_of_std_means: {mean_of_std_means}")

        self.log("mean_of_std_means", mean_of_std_means)

        self.log("val_1_mean", y_ff_hat_mean[0])
        self.log("val_X_mean", y_ff_hat_mean[1])
        self.log("val_2_mean", y_ff_hat_mean[2])

        self.log("val_1_std", y_ff_std_mean[0])
        self.log("val_X_std", y_ff_std_mean[1])
        self.log("val_2_std", y_ff_std_mean[2])

        self.log("val_loss", avg_loss)

        if self.current_epoch % self.plot_every == 0:

            # Make predictions on simulation data
            predictions = []
            for data in self.simulation_data:
                tensor_data = data[1]
                tensor = torch.tensor(
                    tensor_data, dtype=torch.float).to(self.device)
                y_pred = self.forward(tensor)
                y_pred = F.softmax(y_pred, dim=-1)
                predictions.append(y_pred)

            # Make simulation
            # Log cash at the end of simulation and maximum drawdown
            cash, drawdown, all_bets, all_bets_outcomes = self.simulation.run(
                self.simulation_dictionaries, predictions)

            max_drawdown = drawdown.min()
            max_drawdown = min(max_drawdown.items())[1]

            print(f"\nSimulation cash: {cash.iloc[-1]}")
            print(f"Simulation max drawdown: {max_drawdown}")
            print(f"Simulation bets len: {len(all_bets)}")
            # count amount of values 0 in all_bets
            home_bets = all_bets.count(0)
            draw_bets = all_bets.count(1)
            away_bets = all_bets.count(2)
            print(f"Simulation bets home: {home_bets}")
            print(f"Simulation bets draw: {draw_bets}")
            print(f"Simulation bets away: {away_bets}")

            self.log("simulation_cash", cash.iloc[-1])
            self.log("simulation_max_drawdown", max_drawdown)
            self.log("simulation_bets_len", len(all_bets))
            self.log("simulation_bets_home", home_bets)
            self.log("simulation_bets_draw", draw_bets)
            self.log("simulation_bets_away", away_bets)

            def bar_color(df, color1, color2):
                return np.where(df.values > 0, color1, color2).T

            if len(all_bets) > 0:
                # plot cash, drawdown and bets_outcomes
                cash_plot = cash.reset_index(drop=True)
                bets_outcomes_plot = all_bets_outcomes.reset_index(drop=True)
                ax = cash_plot.plot(xlabel='Fixture', ylabel='Cash')
                bets_outcomes_plot.plot(ax=ax, kind='bar', secondary_y=True, color=bar_color(
                    all_bets_outcomes, 'g', 'r')).set_xticklabels([])

                self.logger.experiment.add_figure(
                    "simulation", plt.gcf(), self.current_epoch)
                # self.logger.experiment.add_image("simulation", plt.gcf, self.current_epoch)

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
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        lr_schedulers = {"scheduler": torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=6, gamma=0.99, last_epoch=-1)}
        return [optimizer], [lr_schedulers]
