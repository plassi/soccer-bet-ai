# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pytorch_lightning as pl
from neuralnet import DataModule, LitAutoEncoder

from argparse import ArgumentParser



# %%
# add arguments 
parser = ArgumentParser()
parser.add_argument('--ck_path', default=None, type=str)
parser.add_argument('--hparams_file', default=None, type=str)
parser.add_argument('--test_only', default=None, type=bool)
parser.add_argument('--early_stopping', default=False, type=bool)
parser.add_argument('--gpus', default=0, type=int)
parser.add_argument('--datapath', default='..data/', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--n_workers', default=4, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--min_epochs', default=0, type=int)
parser.add_argument('--max_epochs', default=100, type=int)
args = parser.parse_args()

# Get parameters
print('Load data to get neural network features...')

datamodule = DataModule(batch_size=args.batch_size, n_workers=args.n_workers)
datamodule.prepare_data(datapath=args.datapath)

# init model

if args.ck_path is None:
    model = LitAutoEncoder(datamodule, lr=args.lr)


# Select training or testing loop from arguments

if (args.ck_path is None) and (args.test is None) and (args.gpus == 0) and (args.early_stopping is False):
    # train
    trainer = pl.Trainer(min_epochs=args.min_epochs, max_epochs=args.max_epochs)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

elif (args.ck_path is None) and (args.test is None) and (args.gpus > 0) and (args.early_stopping is False):
    # train
    trainer = pl.Trainer(min_epochs=args.min_epochs, max_epochs=args.max_epochs, gpus=args.gpus, precision=16)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

elif (args.ck_path is not None) and (args.test_only is not None):
    # load model and test it
    model = LitAutoEncoder.load_from_checkpoint(
        checkpoint_path=args.ck_path,
        hparams_file=args.hparams_file,
        datamodule=datamodule,
    )
    trainer = pl.Trainer()
    result = trainer.test(model, datamodule)
