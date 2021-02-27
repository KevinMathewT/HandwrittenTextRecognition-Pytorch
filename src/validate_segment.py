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
        target = self.df.loc[index]['label']

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, target

def parse_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot() 
    labels = []
    for child in root[0]:
        labels.append(child.attrib["text"])
    label = " ".join(labels)
    return label


def create_df():
    forms = glob.glob(config.FORMS_PATH + "/*/*.png")
    df = pd.DataFrame(np.array(forms).reshape(-1, 1), columns=["path"])
    print(df["path", 0])
    df["image_id"] = df.apply(lambda row: row.path.split("\\")[-1].split('.')[0], axis=1)
    print(df)
    df["xml"] = df.apply(lambda row: os.path.join(config.GENERATED_FILES_PATH, "xml") + row.image_id + ".xml", axis=1)
    df["label"] = df.apply(lambda row: parse_xml_file(row.xml), axis=1)
    df = df[["image_id", "path", "label", "xml"]]

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
    net.load_state_dict(torch.load(config.SAVED_WEIGHTS_PATH))
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
    test_pipeline(0, 0, net, loss_fn, dataloader,
                  device, scheduler=None, schd_loss_update=False)

    print(f'Time Taken for Validation: {time.time() - valid_start} seconds |')
    print(f"------------------------------------------------------------------------------")



if __name__ == "__main__":
    df = create_df()
    # validate_model()

    pass
