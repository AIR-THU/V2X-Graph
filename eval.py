from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import TFDDataset
from datasets import V2XTrajDataset
from models.v2x_graph import V2XGraph

import torch
import numpy as np
torch.set_printoptions(threshold=np.inf)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

SUPPORTED_COOPERATION = {
    'V2X-Seq-TFD': ['ego', 'v2i'],
    'V2X-Traj': ['ego', 'v2i', 'v2v', 'v2x']
}


if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True) #V2X-Seq-TFD; V2X-Traj
    parser.add_argument("--cooperation", type=str, required=True) # ego; v2i; v2v; v2x
    parser.add_argument('--default_root_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    
    args = parser.parse_args()

    model = V2XGraph.load_from_checkpoint(checkpoint_path=args.ckpt_path, cooperation=args.cooperation, parallel=False)
    if args.dataset == 'V2X-Seq-TFD':
        val_dataset = TFDDataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
        args.historical_steps = 50
        args.future_steps = 50
    elif args.dataset == 'V2X-Traj':
        val_dataset = V2XTrajDataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
        args.historical_steps = 40
        args.future_steps = 40
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')
    assert args.cooperation in SUPPORTED_COOPERATION[args.dataset], 'Invalid cooperation'
    
    trainer = pl.Trainer.from_argparse_args(args)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer.validate(model, dataloader)
    
