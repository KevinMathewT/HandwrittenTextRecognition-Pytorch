import gc
import time
import glob
import pandas as pd
from joblib import Parallel, delayed
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn

from .engine import get_device, get_net, test_pipeline
from . import config
from .utils import *
from .loss import get_valid_criterion
from .transforms import get_valid_transforms
from .utils import _clean_text
from .segment import segment_lines

import warnings
warnings.filterwarnings("ignore")


class HandWritingFormsDataset(Dataset):
    def __init__(self, df, transforms=None):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        img = get_img(self.df.loc[index]['path']).copy()
        bb = [self.df.loc[index]['x1'], self.df.loc[index]['y1'],
              self.df.loc[index]['x2'], self.df.loc[index]['y2']]
        img = img[bb[1]:bb[3], bb[0]:bb[2], :]
        lines = segment_lines(img)
        target = self.df.loc[index]['label']

        if self.transforms:
            for i, line in enumerate(lines):
                lines[i] = self.transforms(image=line)['image']

        lines = torch.stack(lines)

        return lines, target


def _parse_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []
    for child in root[0]:
        labels.append(child.attrib["text"])
    label = _clean_text(" ".join(labels))
    return label


def _get_bb_of_item(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    character_list = [a for a in root.iter("cmp")]
    if len(character_list) == 0:  # To account for some punctuations that have no words
        return None
    x1 = np.min([int(a.attrib['x']) for a in character_list])
    y1 = np.min([int(a.attrib['y']) for a in character_list])
    x2 = np.max([int(a.attrib['x']) + int(a.attrib['width'])
                 for a in character_list])
    y2 = np.max([int(a.attrib['y']) + int(a.attrib['height'])
                 for a in character_list])

    bb = np.array([x1, y1, x2, y2])
    return bb


def create_df():
    forms = glob.glob(config.FORMS_PATH + "/*/*.png") # Kaggle
    # forms = glob.glob(config.FORMS_PATH + "\*.png")  # PC
    df = pd.DataFrame(np.array(forms).reshape(-1, 1), columns=["path"])
    df["image_id"] = df.apply(
        lambda row: row.path.split("\\")[-1].split('.')[0], axis=1)
    df["xml"] = df.apply(lambda row: os.path.join(
        config.GENERATED_FILES_PATH, "xml") + "/" + row.image_id + ".xml", axis=1)
    df["label"] = df.apply(lambda row: _parse_xml_file(row.xml), axis=1)
    # df[["x1", "x2", "y1", "y2"]] = df.apply(lambda row: _get_bb_of_item(row.xml), axis=1)
    bb = np.array(df.apply(lambda row: _get_bb_of_item(
        row.xml), axis=1).values.tolist())
    bb = pd.DataFrame(bb, columns=["x1", "y1", "x2", "y2"])
    print(bb)
    print(bb.shape)
    df = pd.concat([df, bb], axis=1)
    df = df[["image_id", "path", "label", "xml", "x1", "y1", "x2", "y2"]]
    print(df)
    return df


def get_dataloader():
    dataset = HandWritingFormsDataset(df, transforms=get_valid_transforms())
    return DataLoader(
        dataset,
        batch_size=config.VALID_SEGMENT_BATCH_SIZE,
        drop_last=config.DROP_LAST,
        num_workers=config.CPU_WORKERS,
        shuffle=False)


def validate_model():
    torch.cuda.empty_cache()
    net = get_net(name=config.NET)
    net.load_state_dict(torch.load(config.VALIDATE_WEIGHTS_PATH))
    net.eval()

    print(f"------------------------------------------------------------------------------")
    print(f"Validating Model:            {config.NET}")
    print(f"Image Dimensions:            {config.H}x{config.W}")
    print(f"CNN Backbone:                {config.CNN_BACKBONE}")
    print(f"Mixed Precision Training:    {config.MIXED_PRECISION_TRAIN}")
    print(f"Training Batch Size:         {config.TRAIN_BATCH_SIZE}")
    print(f"Validation Batch Size:       {config.VALID_BATCH_SIZE}")
    print(f"Accumulate Iteration:        {config.ACCUMULATE_ITERATION}")

    dataloader = get_dataloader()
    device = get_device(n=0)
    net = net.to(device)
    scaler = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION_TRAIN else None
    loss_fn = get_valid_criterion(device=device)

    gc.collect()

    valid_start = time.time()
    test_pipeline(net, loss_fn, dataloader, device)

    print(f'Time Taken for Validation: {time.time() - valid_start} seconds |')
    print(f"------------------------------------------------------------------------------")


if __name__ == "__main__":
    df = create_df()
    validate_model()

    pass
