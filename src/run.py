from .dataset import get_loaders, HandWritingLinesDataset
from .transforms import get_train_transforms
from . import config
from tqdm import tqdm

import pandas as pd

from torchvision.utils import save_image

if __name__ == '__main__':
    fold = 0
    # train_folds = pd.read_csv(config.TRAIN_FOLDS)
    # train = train_folds[train_folds.fold != fold]
    # valid = train_folds[train_folds.fold == fold]
    # train_dataset = HandWritingLinesDataset(train, transforms=get_train_transforms())
    
    train_loader, valid_loader          = get_loaders(fold)

    # nimages = 0
    # mean = 0.
    # std = 0.
    # for batch, _ in tqdm(train_loader, total=len(train_loader)):
    #     batch = batch.view(batch.size(0), batch.size(1), -1).cuda()
    #     nimages += batch.size(0)
    #     mean += batch.mean(2).sum(0) 
    #     std += batch.std(2).sum(0)

    # mean /= nimages
    # std /= nimages

    # print(mean)
    # print(std)

    # print(mean, std)

    # for a, b in train_loader:
    #     print(a.size())
    #     print(len(b))
    #     break
