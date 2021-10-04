# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils import data
from neuralnet3 import FootballOddsDecoder
from datamodules import FootballOddsDataModule

from argparse import ArgumentParser


# %%
# add arguments
parser = ArgumentParser()
parser.add_argument('--save_top_k', default=5, type=int)
parser.add_argument('--random_seed', default=None, type=int)
parser.add_argument('--precision_16', default=False, type=bool)
parser.add_argument('--lr_finder', default=False, type=bool)
parser.add_argument('--ck_path', default=None, type=str)
parser.add_argument('--test_only', default=False, type=bool)
parser.add_argument('--early_stopping', default=False, type=bool)
parser.add_argument('--gpus', default=0, type=int)
parser.add_argument('--datapath', default='../data/', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--n_workers', default=4, type=int)
parser.add_argument('--lr', default=2.7e-4, type=float)
parser.add_argument('--min_epochs', default=8, type=int)
parser.add_argument('--max_epochs', default=20, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
args = parser.parse_args()

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
        batch_size=args.batch_size, 
        learning_rate=args.lr, 
        dropout=args.dropout)
elif args.ck_path is not None:
    model = FootballOddsDecoder.load_from_checkpoint(
        checkpoint_path=args.ck_path,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout
    )


checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy",
    save_top_k=args.save_top_k,
    mode="max",
    filename="epoch={epoch:02d}-step={step}-val_acc={val_accuracy:.2f}"
)


# Select training or testing loop from arguments

if args.precision_16 is False:
    # train
    trainer = pl.Trainer(min_epochs=args.min_epochs,
                         max_epochs=args.max_epochs,
                         progress_bar_refresh_rate=4,
                         gpus=args.gpus,
                         profiler="simple",
                         callbacks=[checkpoint_callback])
elif args.precision_16 is True:
    # train
    trainer = pl.Trainer(min_epochs=args.min_epochs,
                         max_epochs=args.max_epochs,
                         progress_bar_refresh_rate=4,
                         gpus=args.gpus,
                         precision=16,
                         profiler="simple",
                         callbacks=[checkpoint_callback])


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
