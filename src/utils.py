import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import cv2
from sklearn.metrics import accuracy_score
from torch.nn.functional import normalize

from torch.utils.data import DataLoader, Dataset

from . import config
from .transforms import *

import editdistance

if config.USE_TPU:
    import torch_xla.core.xla_model as xm


def get_img(path):
    im_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return im_rgb


def stringToClasses(x):
    return torch.tensor(
        [config.CHAR2ID[character] for character in x],
        dtype=torch.long
    )


def bestPathDecoding(x):
    if len(x) == 0:
        return x

    ret = ""
    ret += x[0]

    for i in range(1, len(x)):
        if x[i] != ret[-1]:
            ret += x[i]
    
    ret = ret.replace("~", "")

    return ret

def _clean_text(text):
    text = text.replace("&quot;", "\"")
    text = text.replace("&amp;", "&")
    text = text.replace("\n", "")
    return text


def get_accuracy(predictions, targets, normalize=True):
    predictions = torch.argmax(predictions, dim=1)
    return accuracy_score(targets, predictions, normalize=normalize)


def create_dirs():
    print_fn = print if not config.USE_TPU else xm.master_print
    try:
        os.mkdir(config.WEIGHTS_PATH)
        print_fn(f"Created Folder \'{config.WEIGHTS_PATH}\'")
    except FileExistsError:
        print_fn(f"Folder \'{config.WEIGHTS_PATH}\' already exists.")
    try:
        os.mkdir(os.path.join(config.WEIGHTS_PATH, f'{config.NET}'))
        print_fn(
            f"Created Folder \'{os.path.join(config.WEIGHTS_PATH, f'{config.NET}')}\'")
    except FileExistsError:
        print_fn(
            f"Folder \'{os.path.join(config.WEIGHTS_PATH, f'{config.NET}')}\' already exists.")


class AverageLossMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.curr_batch_avg_loss = 0
        self.avg = 0
        self.running_total_loss = 0
        self.count = 0

    def update(self, curr_batch_avg_loss: float, batch_size: str):
        self.curr_batch_avg_loss = curr_batch_avg_loss
        self.running_total_loss += curr_batch_avg_loss * batch_size
        self.count += batch_size
        self.avg = self.running_total_loss / self.count


class AccuracyMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0
        self.count = 0
        self.sum = 0

    def update(self, y_pred, y_true, batch_size=1):
        self.batch_size = batch_size
        self.count += self.batch_size
        self.score = get_accuracy(y_pred, y_true)
        total_score = self.score * self.batch_size
        self.sum += total_score

    @property
    def avg(self):
        self.avg_score = self.sum/self.count
        return self.avg_score


class EditDistanceMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0
        self.count = 0
        self.sum = 0

    def update(self, y_pred, y_true, batch_size=1):
        self.batch_size = batch_size
        self.count += self.batch_size
        total_score = self.get_avg_edit_distance(y_pred, y_true)
        self.sum += total_score

    def get_avg_edit_distance(self, y_pred, y_true):
        print(y_pred.size(), len(y_true))

        total_distance = 0.0

        for i in len(y_pred):
            pred = y_pred[i].view(config.TIME_STEPS, config.N_CLASSES)
            pred = torch.argmax(pred, 1)
            s = "".join([config.ID2CHAR[id.item()] for id in pred])
            print(s, y_true[i])
            output_decoded = bestPathDecoding(s)
            distance = editdistance.eval(output_decoded, y_true[i])
            total_distance += distance

        return total_distance


    @property
    def avg(self):
        self.avg_score = self.sum/self.count
        return self.avg_score


def freeze_batchnorm_stats(net):
    try:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.eval()
    except ValueError:
        print('error with BatchNorm2d or LayerNorm')
        return
