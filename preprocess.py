from argparse import ArgumentParser

from datamodules import TFDDataModule, V2XTrajDataModule

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True) #V2X-Seq-TFD; V2X-Traj
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)

    args = parser.parse_args()
    
    if args.dataset == 'V2X-Seq-TFD':
        datamodule = TFDDataModule.from_argparse_args(args)
    elif args.dataset == 'V2X-Traj':
        datamodule = V2XTrajDataModule.from_argparse_args(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    datamodule.setup()
