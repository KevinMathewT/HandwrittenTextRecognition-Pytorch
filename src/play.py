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
from .validate_segment import HandWritingFormsDataset, _parse_xml_file, create_df

import warnings
warnings.filterwarnings("ignore")


def display_lines(lines_arr, orient='vertical'):
    plt.figure(figsize=(30, 30))
    if orient not in ['vertical', 'horizontal']:
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


if os.path.isfile(config.FORMS_DF) and False:
    print(f"Loaded cached FORMS_DF from {config.FORMS_DF}")
    df = pd.read_csv(config.FORMS_DF)
else:
    df = create_df()
    df.to_csv(config.FORMS_DF, index=False)

dataset = HandWritingFormsDataset(df, transforms=get_valid_transforms())[:1]
dataloader = DataLoader(
        dataset,
        batch_size=config.VALID_SEGMENT_BATCH_SIZE,
        drop_last=config.DROP_LAST,
        num_workers=config.CPU_WORKERS,
        shuffle=False)

# lines = []
# for line in dataset[0][0]:
#     lines.append(cv2.cvtColor(line.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY))
#     print(cv2.cvtColor(line.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY).shape)

# display_lines(lines)

# print(len(dataset))
# print(len(dataset[0]))
# print(dataset[0][1])
# print(len(dataset[0][0]))

# for i in range(len(dataset[0][0])):
#     plt.imshow(np.transpose(
#         dataset[0][0][i].cpu().detach().numpy(), (1, 2, 0)))
#     plt.show()
# img = get_img("D:\Kevin\Machine Learning\IAM Dataset Full\original\\forms\\a01-000u.png")
# print(img.shape)
# plt.imshow(img)
# print(img)
# plt.show()

net = get_net(name=config.NET)
img = dataset[0][0]
print(img.size())
print(net(img).size())

test_pipeline(net, None, dataloader, get_device(n=0))

