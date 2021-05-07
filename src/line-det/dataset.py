import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A

from .transforms import get_train_transforms, get_valid_transforms
from .create_folds import create_df
from ..utils import get_img, display_image_with_bb
from . import det_config
from .. import config


class LineDetDataset(Dataset):
    def __init__(self, df, transforms_function=None):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms_function = transforms_function

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index):
        records = self.df.loc[index]
        # print("Before Line BB: ", records["line_bb"])
        image_ids = records["image_id"]
        image = get_img(self.df.loc[index]['path']).copy().astype(np.float32)
        full_bb = records["full_bb"][0]
        image /= 255.0

        # display_image_with_bb(image, records["line_bb"])

        # DETR takes in data in COCO format
        bb = records["line_bb"]
        boxes = np.array(bb)
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0] # Height
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1] # Width

        # Area of Bounding Box
        area = boxes[:, 2] * boxes[:, 3]
        area = torch.as_tensor(area, dtype=torch.float32)

        # AS pointed out by PRVI It works better if the main class is labelled as zero
        labels = np.zeros(len(boxes), dtype=np.int32)

        if self.transforms_function:
            transforms = self.transforms_function(bb=full_bb)
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']

        # Normalizing BBOXES

        _, h, w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'], rows=h, cols=w)
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.long),
            'image_id': torch.tensor([index]), 'area': area
        }

        # print("After Line BB: ", records["line_bb"])
        # print("Normalized BB: ", boxes)
        return image, target, image_ids


def collate_fn(batch):
    return tuple(zip(*batch))


def get_train_dataloader(train):
    train_dataset = LineDetDataset(train, transforms_function=get_train_transforms)
    return DataLoader(
        train_dataset,
        batch_size=det_config.TRAIN_BATCH_SIZE,
        drop_last=det_config.DROP_LAST,
        num_workers=det_config.CPU_WORKERS,
        collate_fn=collate_fn,
        shuffle=True)


def get_valid_dataloader(valid):
    valid_dataset = LineDetDataset(valid, transforms_function=get_valid_transforms)
    return DataLoader(
        valid_dataset,
        batch_size=det_config.VALID_BATCH_SIZE,
        drop_last=det_config.DROP_LAST,
        num_workers=det_config.CPU_WORKERS,
        collate_fn=collate_fn,
        shuffle=False)


def get_loaders(fold):
    train_folds = create_df()
    train = train_folds[train_folds.fold != fold]
    valid = train_folds[train_folds.fold == fold]

    train_loader = get_train_dataloader(
        train.drop(['fold'], axis=1))
    valid_loader = get_valid_dataloader(
        valid.drop(['fold'], axis=1))

    return train_loader, valid_loader


if __name__ == "__main__":
    df = create_df()
    dataset = LineDetDataset(df=df, transforms_function=get_train_transforms)
    for image, target, image_id in dataset:
        print(image.shape)
        print(target)
        # display_image_with_bb(image.permute(1, 2, 0).numpy(), target["boxes"].numpy().astype(np.int32), scale=1, format="coco")
        break
