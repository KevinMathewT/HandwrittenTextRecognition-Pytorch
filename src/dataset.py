import os
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

# from fmix import sample_mask, make_low_freq_image, binarise_mask

from . import config
from .utils import *
from .transforms import *

if config.USE_TPU:
    import torch_xla.core.xla_model as xm

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


class HandWritingLinesDataset(Dataset):
    def __init__(self, df, transforms=None):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        img = get_img(self.df.loc[index]['path']).copy()
        target = self.df.loc[index]['label']

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, target

import torchvision
import matplotlib.pyplot as plt

def get_train_dataloader(train):
    train_dataset = HandWritingLinesDataset(train, transforms=get_train_transforms())
    a, b = train_dataset[3]
    print(a, b)
    # torchvision.utils.save_image(a, "dataloader.png")
    plt.imshow(a.permute(1, 2, 0))
    plt.show()
    if config.USE_TPU:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(), # divide dataset among this many replicas
            rank=xm.get_ordinal(), # which replica/device/core
            shuffle=True)
        return DataLoader(
            train_dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            sampler=train_sampler,
            num_workers=config.CPU_WORKERS,
            drop_last=config.DROP_LAST)
    else:
        return DataLoader(
            train_dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            drop_last=config.DROP_LAST,
            num_workers=config.CPU_WORKERS,
            shuffle=True)


def get_valid_dataloader(valid):
    valid_dataset = HandWritingLinesDataset(valid, transforms=get_valid_transforms())
    if config.USE_TPU:
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)
        return DataLoader(
            valid_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            sampler=valid_sampler,
            num_workers=config.CPU_WORKERS,
            drop_last=config.DROP_LAST)
    else:
        return DataLoader(
            valid_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            drop_last=config.DROP_LAST,
            num_workers=config.CPU_WORKERS,
            shuffle=False)


def get_loaders(fold):
    train_folds = pd.read_csv(config.TRAIN_FOLDS)
    train = train_folds[train_folds.fold != fold]
    valid = train_folds[train_folds.fold == fold]

    train_loader = get_train_dataloader(
        train.drop(['fold'], axis=1))
    valid_loader = get_valid_dataloader(
        valid.drop(['fold'], axis=1))

    return train_loader, valid_loader