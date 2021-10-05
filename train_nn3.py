# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch._C import has_cuda
from neuralnet3 import FootballOddsDecoder
from datamodules import FootballOddsDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from argparse import ArgumentParser


# %%
# add arguments
parser = ArgumentParser()
parser.add_argument('--h_layers', default=1, type=int)
parser.add_argument('--h_features', default=1024, type=int)
parser.add_argument('--save_top_k', default=5, type=int)
parser.add_argument('--random_seed', default=None, type=int)
parser.add_argument('--precision_16', default=False, type=bool)
parser.add_argument('--lr_finder', default=False, type=bool)
parser.add_argument('--ck_path', default=None, type=str)
parser.add_argument('--test_only', default=False, type=bool)
# parser.add_argument('--early_stopping', default=False, type=bool)
parser.add_argument('--gpus', default=0, type=int)
parser.add_argument('--datapath', default='../data/', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--n_workers', default=4, type=int)
parser.add_argument('--lr', default=2.7e-6, type=float)
parser.add_argument('--min_epochs', default=0, type=int)
parser.add_argument('--max_epochs', default=20, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
args = parser.parse_args()


# %%
# Early stoppers
early_stop_callback_1 = EarlyStopping(
    monitor="val_accuracy", min_delta=0.0001, divergence_threshold=0.000001, patience=100, verbose=True, mode="max")


checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy",
    save_top_k=args.save_top_k,
    save_last=True,
    mode="max",
    filename="{epoch:02d}-{step}-{val_accuracy:.2f}",
    verbose=True,
)


# Get parameters
print('Load data to get neural network features...')

datamodule = FootballOddsDataModule(
    batch_size=args.batch_size,
    n_workers=args.n_workers,
    random_seed=args.random_seed,)
datamodule.prepare_data(datapath=args.datapath)


# init model

if args.ck_path is None:
    model = FootballOddsDecoder(
        h_layers=args.h_layers,
        h_features=args.h_features,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout)
elif args.ck_path is not None:
    model = FootballOddsDecoder.load_from_checkpoint(
        checkpoint_path=args.ck_path,
    )



# Select training or testing loop from arguments

if args.precision_16 is False:
    # train
    trainer = pl.Trainer(min_epochs=args.min_epochs,
                         max_epochs=args.max_epochs,
                         progress_bar_refresh_rate=4,
                         gpus=args.gpus,
                         #  profiler="simple",
                         stochastic_weight_avg=True,
                         callbacks=[checkpoint_callback, early_stop_callback_1])
elif args.precision_16 is True:
    # train
    trainer = pl.Trainer(min_epochs=args.min_epochs,
                         max_epochs=args.max_epochs,
                         progress_bar_refresh_rate=4,
                         gpus=args.gpus,
                         precision=16,
                         #  profiler="simple",
                         stochastic_weight_avg=True,
                         callbacks=[checkpoint_callback, early_stop_callback_1])


# Learning_rate finder
if args.lr_finder is True:

    # Run learning rate finder

    lr_finder = trainer.tuner.lr_find(model, datamodule)

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    print("\nlr_finder.suggestion(): ", new_lr)

    # update hparams of the model

    if args.ck_path is None:
        model = FootballOddsDecoder(
            batch_size=args.batch_size, learning_rate=new_lr, dropout=args.dropout)
    elif args.ck_path is not None:
        model = FootballOddsDecoder.load_from_checkpoint(
            checkpoint_path=args.ck_path,
            hparams_file=args.hparams_file,
            batch_size=args.batch_size,
            learning_rate=new_lr,
            dropout=args.dropout
        )


# Fit model

if not args.test_only:
    trainer.fit(model, datamodule)

elif args.test_only:
    trainer.test(model, datamodule)
