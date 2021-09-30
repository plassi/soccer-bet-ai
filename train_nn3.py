# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neuralnet3 import FootballOddsDecoder
from datamodules import FootballOddsDataModule

from argparse import ArgumentParser


# %%
# add arguments
parser = ArgumentParser()
parser.add_argument('--ck_path', default=None, type=str)
parser.add_argument('--hparams_file', default=None, type=str)
parser.add_argument('--test_only', default=False, type=bool)
parser.add_argument('--early_stopping', default=False, type=bool)
parser.add_argument('--gpus', default=0, type=int)
parser.add_argument('--datapath', default='../data/', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--n_workers', default=4, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--min_epochs', default=0, type=int)
parser.add_argument('--max_epochs', default=1000, type=int)
parser.add_argument('--dropout', default=0.2, type=float)
args = parser.parse_args()

# Get parameters
print('Load data to get neural network features...')

datamodule = FootballOddsDataModule(
    batch_size=args.batch_size, n_workers=args.n_workers)
datamodule.prepare_data(datapath=args.datapath)

# init model

if args.ck_path is None:
    model = FootballOddsDecoder(
        batch_size=args.batch_size, learning_rate=args.lr, dropout=args.dropout)
elif args.ck_path is not None:
    model = FootballOddsDecoder.load_from_checkpoint(
        checkpoint_path=args.ck_path,
        hparams_file=args.hparams_file,
        datamodule=datamodule,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout
    )


checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=20,
    mode="min",
)


# Select training or testing loop from arguments

if (args.test_only is False) and (args.early_stopping is False):
    # train
    trainer = pl.Trainer(min_epochs=args.min_epochs,
                         max_epochs=args.max_epochs,
                         progress_bar_refresh_rate=4,
                         gpus=args.gpus,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule)

elif (args.test_only is False) and (args.early_stopping is False):
    # load model and test it

    trainer = pl.Trainer(min_epochs=args.min_epochs,
                         max_epochs=args.max_epochs,
                         progress_bar_refresh_rate=4,
                         gpus=args.gpus,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule)
