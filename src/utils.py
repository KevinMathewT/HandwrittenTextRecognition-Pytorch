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
from .decoder import bestPathDecoder, beamSearchDecoder

import editdistance
import jiwer

if config.DECODER == "BestPathDecoder":
    decoding_fn = bestPathDecoder
elif config.DECODER == "BeamSearchDecoder":
    decoding_fn = beamSearchDecoder


def get_img(path):
    im_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return im_rgb


def stringToClasses(x):
    return torch.tensor(
        [config.CHAR2ID[character] for character in x],
        dtype=torch.long
    )


def _clean_text(text):
    text = text.replace("&quot;", "\"")
    text = text.replace("&amp;", "&")
    text = text.replace("\n", "")
    return text


def get_accuracy(predictions, targets, normalize=True):
    predictions = torch.argmax(predictions, dim=1)
    return accuracy_score(targets, predictions, normalize=normalize)


def create_dirs():
    try:
        os.mkdir(config.WEIGHTS_PATH)
        print(f"Created Folder \'{config.WEIGHTS_PATH}\'")
    except FileExistsError:
        print(f"Folder \'{config.WEIGHTS_PATH}\' already exists.")
    try:
        os.mkdir(os.path.join(config.WEIGHTS_PATH, f'{config.NET}'))
        print(
            f"Created Folder \'{os.path.join(config.WEIGHTS_PATH, f'{config.NET}')}\'")
    except FileExistsError:
        print(
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


class StringMatchingMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0
        self.count = 0
        self.ed = 0
        self.wer = 0
        self.mer = 0
        self.wil = 0

    def update(self, y_pred, y_true, batch_size=1):
        self.count += batch_size
        total_ed = self.get_total_edit_distance(y_pred, y_true)
        total_wer, total_mer, total_wil = self.get_total_error_rates(y_pred, y_true)
        self.ed += total_ed
        self.wer += total_wer
        self.mer += total_mer
        self.wil += total_wil

    def update_with_strings(self, y_pred, y_true, batch_size=1):
        self.count += batch_size
        total_ed = self.get_total_edit_distance_with_strings(y_pred, y_true)
        total_wer, total_mer, total_wil = self.get_total_error_rates_with_strings(y_pred, y_true)
        self.ed += total_ed
        self.wer += total_wer
        self.mer += total_mer
        self.wil += total_wil

    @staticmethod
    def get_total_edit_distance(y_pred, y_true):
        # print(y_pred.size(), len(y_true))

        y_pred = y_pred.permute(1, 0, 2)
        total_distance = 0.0

        for i in range(len(y_true)):
            pred = y_pred[i].view(-1, config.N_CLASSES)
            output_decoded = decoding_fn(pred.detach().cpu().numpy())
            distance = editdistance.eval(output_decoded, y_true[i])
            # print(output_decoded, y_true[i], distance)
            total_distance += distance

        return total_distance

    @staticmethod
    def get_total_error_rates(y_pred, y_true):
        # print(y_pred.size(), len(y_true))

        y_pred = y_pred.permute(1, 0, 2)
        total_wer = 0.0
        total_mer = 0.0
        total_wil = 0.0

        for i in range(len(y_true)):
            pred = y_pred[i].view(-1, config.N_CLASSES)
            output_decoded = decoding_fn(pred.detach().cpu().numpy())
            error = jiwer.compute_measures(y_true[i], output_decoded)
            # print(output_decoded, y_true[i], distance)
            total_wer += error['wer']
            total_mer += error['mer']
            total_wil += error['wil']

        return total_wer, total_mer, total_wil

    @staticmethod
    def get_total_edit_distance_with_strings(y_pred, y_true):
        total_distance = 0.0

        for i in range(len(y_true)):
            distance = editdistance.eval(y_pred[i], y_true[i])
            total_distance += distance

        return total_distance

    @staticmethod
    def get_total_error_rates_with_strings(y_pred, y_true):
        total_wer = 0.0
        total_mer = 0.0
        total_wil = 0.0

        for i in range(len(y_true)):
            error = jiwer.compute_measures(y_true[i], y_pred[i])
            total_wer += error['wer']
            total_mer += error['mer']
            total_wil += error['wil']

        return total_wer, total_mer, total_wil

    @property
    def avg_edit_distance(self):
        self.avg_score = self.ed / self.count
        return self.avg_score

    @property
    def avg_wer(self):
        self.avg_score = self.wer / self.count
        return self.avg_score

    @property
    def avg_mer(self):
        self.avg_score = self.mer / self.count
        return self.avg_score

    @property
    def avg_wil(self):
        self.avg_score = self.wil / self.count
        return self.avg_score


def get_one_from_batch(y_pred, y_true):
    y_pred = y_pred.permute(1, 0, 2)
    i = random.randint(0, len(y_pred) - 1)
    s = decoding_fn(y_pred[i].detach().cpu().numpy())
    return s, y_true[i]


def freeze_batchnorm_stats(net):
    try:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.eval()
    except ValueError:
        print('error with BatchNorm2d or LayerNorm')
        return
