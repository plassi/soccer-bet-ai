# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lstm import FootballOddsLSTM
from lstm_datamodule import FootballOddsDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from argparse import ArgumentParser

import os
import pickle


# %%
# add arguments
parser = ArgumentParser()
parser.add_argument('--cache', default=False, type=bool)
parser.add_argument('--h_layers', default=3, type=int)
parser.add_argument('--h_features', default=256, type=int)
parser.add_argument('--save_top_k', default=3, type=int)
parser.add_argument('--random_seed', default=None, type=int)
parser.add_argument('--precision_16', default=False, type=bool)
parser.add_argument('--ck_path', default=None, type=str)
# parser.add_argument('--early_stopping', default=False, type=bool)
parser.add_argument('--gpus', default=0, type=int)
parser.add_argument('--datapath', default='../data_test/', type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--n_workers', default=4, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--min_epochs', default=0, type=int)
parser.add_argument('--max_epochs', default=20, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
args = parser.parse_args()


# %%
# Early stoppers

early_stop_callback = EarlyStopping(
    monitor="val_accuracy", min_delta=0.0001, patience=10, verbose=True, mode="max")


checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy",
    save_top_k=args.save_top_k,
    save_last=True,
    mode="max",
    filename="{epoch:02d}-{step}-{val_accuracy:.6f}",
    verbose=True,
)


# Get parameters
print('Loading datamodule...')

# if file data_module.pkl exists, load it
if os.path.isfile(args.datapath + 'data_module.pkl') and args.cache == True:
    print("load from cache")
    with open(args.datapath + 'data_module.pkl', 'rb') as f:
        print(f)
        datamodule = pickle.load(f)
else:
    print('Create datamodule')
    datamodule = FootballOddsDataModule(
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        random_seed=args.random_seed,)
    datamodule.prepare_data(datapath=args.datapath)
    if args.cache == True:
        with open(args.datapath + 'data_module.pkl', 'wb') as f:
            pickle.dump(datamodule, f)


# init model

input_features = datamodule.dataset.X_data.shape[1]

if args.ck_path is None:
    model = FootballOddsLSTM(
        h_layers=args.h_layers,
        h_features=args.h_features,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        input_features=input_features,)
elif args.ck_path is not None:
    model = FootballOddsLSTM.load_from_checkpoint(
        checkpoint_path=args.ck_path,
    )


# Select training or testing loop from arguments

if args.precision_16 is False:
    # train
    trainer = pl.Trainer(min_epochs=args.min_epochs,
                         max_epochs=args.max_epochs,
                         progress_bar_refresh_rate=20,
                         gpus=args.gpus,
                         #  profiler="simple",
                         stochastic_weight_avg=True,
                         callbacks=[checkpoint_callback, early_stop_callback])
elif args.precision_16 is True:
    # train
    trainer = pl.Trainer(min_epochs=args.min_epochs,
                         max_epochs=args.max_epochs,
                         progress_bar_refresh_rate=20,
                         gpus=args.gpus,
                         precision=16,
                         #  profiler="simple",
                         stochastic_weight_avg=True,
                         callbacks=[checkpoint_callback, early_stop_callback])


# Fit model

trainer.fit(model, datamodule)
