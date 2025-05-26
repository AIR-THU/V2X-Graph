import torch
import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import TFDDataModule, V2XTrajDataModule
from models.v2x_graph import V2XGraph

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

SUPPORTED_COOPERATION = {
    'V2X-Seq-TFD': ['ego', 'v2i'],
    'V2X-Traj': ['ego', 'v2i', 'v2v', 'v2x']
}


if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='')
    parser.add_argument("--dataset", type=str, required=True) #V2X-Seq-TFD; V2X-Traj
    parser.add_argument("--cooperation", type=str, required=True) # ego; v2i; v2v; v2x
    parser.add_argument('--default_root_dir', type=str, default='')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    parser.add_argument('--accumulate_grad_batches', type=int, default=16)
    parser = V2XGraph.add_model_specific_args(parser)
    args = parser.parse_args()
    
    if args.dataset == 'V2X-Seq-TFD':
        datamodule = TFDDataModule.from_argparse_args(args)
        args.historical_steps = 50
        args.future_steps = 50
    elif args.dataset == 'V2X-Traj':
        datamodule = V2XTrajDataModule.from_argparse_args(args)
        args.historical_steps = 40
        args.future_steps = 40
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    assert args.cooperation in SUPPORTED_COOPERATION[args.dataset], "Invalid cooperation"
    
    datamodule.setup()
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, accelerator='gpu', callbacks=[model_checkpoint])
    model = V2XGraph(**vars(args))

    trainer.fit(model, datamodule)
