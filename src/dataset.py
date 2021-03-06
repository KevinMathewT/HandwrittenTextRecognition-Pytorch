import pandas as pd

import torchvision

from .transforms import *
from .utils import *


# from fmix import sample_mask, make_low_freq_image, binarise_mask


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
        image = get_img(self.df.loc[index]['path']).copy()
        target = self.df.loc[index]['label']

        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, target


def get_train_dataloader(train):
    train_dataset = HandWritingLinesDataset(train, transforms=get_train_transforms())
    # a, b = train_dataset[3]
    # print(a, b)
    # plt.imshow(a.permute(1, 2, 0))
    # plt.show()
    # torchvision.utils.save_image(a, "dataloader.png")
    return DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        drop_last=config.DROP_LAST,
        num_workers=config.CPU_WORKERS,
        shuffle=True)


def get_valid_dataloader(valid):
    valid_dataset = HandWritingLinesDataset(valid, transforms=get_valid_transforms())
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
