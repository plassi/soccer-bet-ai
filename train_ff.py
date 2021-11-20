# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from ff import FootballOddsFF
from datamodule_ff import FootballOddsDataModule
from pytorch_lightning.callbacks import LearningRateMonitor

from argparse import ArgumentParser

import os
import pickle


# %%
# add arguments
parser = ArgumentParser()
parser.add_argument('--feature_set', default=2, type=int)
parser.add_argument('--cache', default=True, type=bool)
parser.add_argument('--h_layers', default=3, type=int)
parser.add_argument('--h_features', default=1024, type=int)
parser.add_argument('--save_top_k', default=1, type=int)
parser.add_argument('--random_seed', default=None, type=int)
parser.add_argument('--precision_16', default=False, type=bool)
parser.add_argument('--ck_path', default=None, type=str)
# parser.add_argument('--early_stopping', default=False, type=bool)
parser.add_argument('--gpus', default=0, type=int)
parser.add_argument('--datapath', default='../data_ff', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--n_workers', default=8, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--min_epochs', default=0, type=int)
parser.add_argument('--max_epochs', default=4000, type=int)
parser.add_argument('--dropout', default=0.75, type=float)
args = parser.parse_args()


if args.random_seed is not None:
    pl.seed_everything(args.random_seed)

# %%
# Early stoppers

# early_stop_callback = EarlyStopping(
#     monitor="val_loss", min_delta=0.0001, patience=int(args.max_epochs / 6), verbose=True, mode="min")
lr_monitor = LearningRateMonitor(logging_interval='epoch')

checkpoint_callback_0 = ModelCheckpoint(
    monitor="simulation_cash_max",
    save_top_k=args.save_top_k,
    save_last=True,
    mode="max",
    filename="{epoch:02d}-{step}-{simulation_cash_max:.2f}-{simulation_cash_end:.2f}-{simulation_max_drawdown:.2f}",
    verbose=True,
    every_n_epochs=1
)

checkpoint_callback_1 = ModelCheckpoint(
    monitor="simulation_cash_end",
    save_top_k=args.save_top_k,
    save_last=False,
    mode="max",
    filename="{epoch:02d}-{step}-{simulation_cash_max:.2f}-{simulation_cash_end:.2f}-{simulation_max_drawdown:.2f}",
    verbose=True,
    every_n_epochs=1
)

checkpoint_callback_2 = ModelCheckpoint(
    monitor="simulation_max_drawdown",
    save_top_k=args.save_top_k,
    save_last=False,
    mode="max",
    filename="{epoch:02d}-{step}-{simulation_cash_max:.2f}-{simulation_cash_end:.2f}-{simulation_max_drawdown:.2f}",
    verbose=True,
    every_n_epochs=1
)


# Get parameters
print('Create datamodule...')

datamodule = FootballOddsDataModule(
    batch_size=args.batch_size,
    n_workers=args.n_workers,
    random_seed=args.random_seed,)
datamodule.prepare_data(datapath=args.datapath,
                        feature_set=args.feature_set,
                        cache=args.cache)


# init model
print('Init model...')
print('datamodule_' + str(args.feature_set))
print("datamodule.dataset.X_data.shape", datamodule.dataset.X_data.shape)
input_features = datamodule.dataset.X_data.shape[1]

if args.ck_path is None:
    model = FootballOddsFF(
        h_layers=args.h_layers,
        h_features=args.h_features,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        input_features=input_features,
        datapath=args.datapath,
        feature_set=args.feature_set,)
elif args.ck_path is not None:
    model = FootballOddsFF.load_from_checkpoint(
        checkpoint_path=args.ck_path,
    )


# Create logger
# logger = TensorBoardLogger("lightning_logs", log_graph=True)


# Select training or testing loop from arguments

if args.precision_16 is False:
    # train
    trainer = pl.Trainer(  # logger=logger,
        accumulate_grad_batches=1,
        gradient_clip_val=0.5,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        progress_bar_refresh_rate=20,
        gpus=args.gpus,
        #  profiler="simple",
        stochastic_weight_avg=True,
        callbacks=[checkpoint_callback_0, checkpoint_callback_1, checkpoint_callback_2, lr_monitor],)
elif args.precision_16 is True:
    # train
    trainer = pl.Trainer(  # logger=logger,
        accumulate_grad_batches=1,
        gradient_clip_val=0.5,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        progress_bar_refresh_rate=20,
        gpus=args.gpus,
        precision=16,
        #  profiler="simple",
        stochastic_weight_avg=True,
        callbacks=[checkpoint_callback_0, checkpoint_callback_1, checkpoint_callback_2, lr_monitor],)


# Fit model

trainer.fit(model, datamodule)
