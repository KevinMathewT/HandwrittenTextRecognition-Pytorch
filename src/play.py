import gc
import time
import glob
import pandas as pd
from joblib import Parallel, delayed
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from .engine import get_device, get_net, test_pipeline
from . import config
from .utils import *
from .loss import get_valid_criterion
from .transforms import get_valid_transforms
from .utils import _clean_text
from .segment import segment_lines
from .validate_segment import HandWritingFormsDataset, parse_xml_file

import warnings
warnings.filterwarnings("ignore")

def display_lines(lines_arr, orient='vertical'):
    plt.figure(figsize=(30, 30))
    if not orient in ['vertical', 'horizontal']:
        raise ValueError(
            "Orientation is on of 'vertical', 'horizontal', defaul = 'vertical'")
    if orient == 'vertical':
        for i, l in enumerate(lines_arr):
            line = l
            plt.subplot(2, 10, i+1)  # A grid of 2 rows x 10 columns
            plt.axis('off')
            plt.title("Line #{0}".format(i))
            _ = plt.imshow(line, cmap='gray', interpolation='bicubic')
            # to hide tick values on X and Y axis
            plt.xticks([]), plt.yticks([])
    else:
        for i, l in enumerate(lines_arr):
            line = l
            plt.subplot(40, 1, i+1)  # A grid of 40 rows x 1 columns
            plt.axis('off')
            plt.title("Line #{0}".format(i))
            _ = plt.imshow(line, cmap='gray', interpolation='bicubic')
            # to hide tick values on X and Y axis
            plt.xticks([]), plt.yticks([])
    plt.show()

def create_df():
    forms = glob.glob("D:\Kevin\Machine Learning\IAM Dataset Full\original\\forms/*.png")
    df = pd.DataFrame(np.array(forms).reshape(-1, 1), columns=["path"])
    print(df)
    df["image_id"] = df.apply(lambda row: row.path.split("\\")[-1].split('.')[0], axis=1)
    print(df)
    df["xml"] = df.apply(lambda row: os.path.join(config.GENERATED_FILES_PATH, "xml") + "/" + row.image_id + ".xml", axis=1)
    df["label"] = df.apply(lambda row: parse_xml_file(row.xml), axis=1)
    df = df[["image_id", "path", "label", "xml"]]

    print(df["label"])
    return df

if os.path.isfile(config.FORMS_DF):
    print(f"Loaded cached FORMS_DF from {config.FORMS_DF}")
    df = pd.read_csv(config.FORMS_DF)
else:
    df = create_df()
    df.to_csv(config.FORMS_DF, index=False)

dataset = HandWritingFormsDataset(df, transforms=get_valid_transforms())

# lines = []
# for line in dataset[0][0]:
#     lines.append(cv2.cvtColor(line.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY))
#     print(cv2.cvtColor(line.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY).shape)

# display_lines(lines)

print(dataset[0][0][0].permute(1, 2, 0).size())
plt.imshow(dataset[0][0][0].permute(1, 2, 0).numpy() / 256)
plt.show()
img = get_img("D:\Kevin\Machine Learning\IAM Dataset Full\original\\forms\\a01-000u.png")
print(img.shape)
plt.imshow(img)
plt.show()