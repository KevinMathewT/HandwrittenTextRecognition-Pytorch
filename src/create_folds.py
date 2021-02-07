import os
from numpy.core.fromnumeric import clip
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import seaborn as sns

from . import config
from .utils import _clean_text

def read_file(file):
    f = open(file, "r")
    paths = []
    labels = []
    for line in f.readlines():
        path = line.split('\t')[0]
        label = _clean_text('\t'.join(line.split('\t')[1:]))
        if len(label) >= config.MIN_LEN_ALLOWED and len(label) <= config.MAX_LEN_ALLOWED:
            paths.append(path)
            labels.append(label)
    paths = np.array(paths).reshape(-1, 1)
    labels = np.array(labels).reshape(-1, 1)

    df = pd.DataFrame(np.concatenate([paths, labels], axis=1), columns=["path", "label"])

    return df

if __name__ == "__main__":
    train_1 = read_file(config.TRAIN_1)
    train_1["image_id"] = train_1.apply(
        lambda row: row.path.split("/")[-1].split('.')[0], axis=1)
    train_1["path"] = train_1.apply(lambda row: os.path.join(
        config.TRAIN_IMAGES_DIR_1, row.path.split("/")[-1]), axis=1)

    train_2 = read_file(config.TRAIN_2)
    train_2["image_id"] = train_2.apply(
        lambda row: row.path.split("/")[-1].split('.')[0], axis=1)
    train_2["path"] = train_2.apply(lambda row: os.path.join(
        config.TRAIN_IMAGES_DIR_2, row.path.split("/")[-1]), axis=1)

    train = pd.concat([train_1, train_2], axis=0)
    train["length"] = train.apply(lambda row: len(row.label), axis=1)
    train["length_bin"] = train.apply(lambda row: (len(row.label) // config.LENGTH_BIN_SIZE) * config.LENGTH_BIN_SIZE, axis=1)
    train = train[["image_id", "label", "path", "length", "length_bin"]].reset_index(drop=True)
    
    train["fold"] = -1
    skf = StratifiedKFold(n_splits=config.FOLDS, shuffle=True, random_state=config.SEED)
    X = train['image_id']
    y = train["length_bin"]

    plot = sns.displot(train["length"])
    plot.savefig("length_distribution_plot.png")

    print(f"Training Data Dimensions: {train.shape}")

    for fold, (train_index, test_index) in tqdm(enumerate(skf.split(X, y)), total=config.FOLDS):
        train.loc[test_index, "fold"] = fold

    train = train[["image_id", "label", "path", "fold"]].reset_index(drop=True)
    train.to_csv(config.TRAIN_FOLDS, index=False)
    print(f"{config.FOLDS} folds created and saved at: {config.TRAIN_FOLDS}.")