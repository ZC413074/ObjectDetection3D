from DataUtils.modelnet_datasets import *
import torch


def create_dataloader(args, mode):
    dataset = ModelNetDataset(args, mode, args.process_data)
    batch_size = min(args.batch_size, len(dataset))
    drop_last = True if mode == 'train' else False
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             drop_last=drop_last)
    return dataloader, dataset
