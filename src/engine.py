import time
from src.utils import get_accuracy
from sklearn.metrics import accuracy_score
import numpy as np

import torch
from torch import optim
from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief

from . import config
from .models.models import *
from .utils import StringMatchingMetrics, AverageLossMeter, get_one_from_batch


def train_one_epoch(fold, epoch, model, loss_fn, optimizer, train_loader, device, scaler, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = AverageLossMeter()
    running_string_metrics = StringMatchingMetrics()
    total_steps = len(train_loader)
    pbar = enumerate(train_loader)
    optimizer.zero_grad()
    preds, trues = [], []

    for step, (imgs, image_labels) in pbar:
        imgs, image_labels = imgs.to(device, dtype=torch.float32), image_labels
        curr_batch_size = imgs.size(0)

        # print(image_labels.shape, exam_label.shape)
        if config.MIXED_PRECISION_TRAIN:
            with torch.cuda.amp.autocast():
                image_preds = model(imgs)
                print(image_preds.size())
                loss = loss_fn(image_preds, image_labels)
                # print(image_preds.size(), loss)
                running_loss.update(
                    curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)
                running_string_metrics.update(
                    y_pred=image_preds.detach().cpu(),
                    y_true=image_labels,
                    batch_size=curr_batch_size)

            scaler.scale(loss).backward()

            if ((step + 1) % config.ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step(epoch + (step / total_steps))

        else:
            image_preds = model(imgs)
            loss = loss_fn(image_preds, image_labels)
            loss.backward()

            if ((step + 1) % config.ACCUMULATE_ITERATION == 0) or ((step + 1) == total_steps):
                optimizer.step()
                if scheduler is not None and schd_batch_update:
                    scheduler.step(epoch + (step / total_steps))
                optimizer.zero_grad()

            running_loss.update(
                curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)
            running_string_metrics.update(
                y_pred=image_preds.detach().cpu(),
                y_true=image_labels,
                batch_size=curr_batch_size)

            # print("Loss Update:", running_loss.avg)
            # print("Acc Update:", running_loss.avg)

        loss = running_loss.avg
        edit = running_string_metrics.avg_edit_distance
        wer = running_string_metrics.avg_wer
        mer = running_string_metrics.avg_mer
        wil = running_string_metrics.avg_wil
        pred, true = get_one_from_batch(image_preds, image_labels)
        preds.append(pred)
        trues.append(true)

        if ((config.LEARNING_VERBOSE and (step + 1) % config.VERBOSE_STEP == 0)) or ((step + 1) == total_steps) or ((step + 1) == 1):
            description = f'[{fold}/{config.FOLDS - 1}][{epoch:>2d}/{config.MAX_EPOCHS - 1:>2d}][{step + 1:>4d}/{total_steps:>4d}] Loss: {loss:.4f} | ED: {edit:.4f} | WER: {wer:.4f} | Time: {(time.time() - t) / 60:.2f} m'
            print(description, flush=True)

        if config.DEBUG_MODE: break

    c = 0
    for pred, true in zip(preds, trues):
        print(f"------------------------------------------------------------------------------")
        print(f"Predicted: {pred}")
        print(f"Expected:  {true}")
        c += 1
        if c >= 5: break
        
    print(f"------------------------------------------------------------------------------")
    
    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(fold, epoch, model, loss_fn, valid_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    t = time.time()
    running_loss = AverageLossMeter()
    running_string_metrics = StringMatchingMetrics()
    total_steps = len(valid_loader)
    pbar = enumerate(valid_loader)
    preds, trues = [], []

    for step, (imgs, image_labels) in pbar:
        imgs, image_labels = imgs.to(device, dtype=torch.float32), image_labels
        curr_batch_size = imgs.size(0)

        # print(image_labels.shape, exam_label.shape)
        if config.MIXED_PRECISION_TRAIN:
            with torch.cuda.amp.autocast():
                image_preds = model(imgs)
                loss = loss_fn(image_preds, image_labels)
                # print(image_preds.size(), loss)
                running_loss.update(
                    curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)
                running_string_metrics.update(
                    y_pred=image_preds.detach().cpu(),
                    y_true=image_labels,
                    batch_size=curr_batch_size)

        else:
            image_preds = model(imgs)
            loss = loss_fn(image_preds, image_labels)

            running_loss.update(
                curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)
            running_string_metrics.update(
                y_pred=image_preds.detach().cpu(),
                y_true=image_labels,
                batch_size=curr_batch_size)

            # print("Loss Update:", running_loss.avg)
            # print("Acc Update:", running_loss.avg)

        loss = running_loss.avg
        edit = running_string_metrics.avg_edit_distance
        wer = running_string_metrics.avg_wer
        mer = running_string_metrics.avg_mer
        wil = running_string_metrics.avg_wil
        pred, true = get_one_from_batch(image_preds, image_labels)
        preds.append(pred)
        trues.append(true)


        if ((config.LEARNING_VERBOSE and (step + 1) % config.VERBOSE_STEP == 0)) or ((step + 1) == total_steps) or ((step + 1) == 1):
            description = f'[{fold}/{config.FOLDS - 1}][{epoch:>2d}/{config.MAX_EPOCHS - 1:>2d}][{step + 1:>4d}/{total_steps:>4d}] Loss: {loss:.4f} | ED: {edit:.4f} | WER: {wer:.4f} | Time: {(time.time() - t) / 60:.2f} m'
            print(description, flush=True)
            
        if config.DEBUG_MODE: break

    c = 0
    for pred, true in zip(preds, trues):
        print(f"------------------------------------------------------------------------------")
        print(f"Predicted: {pred}")
        print(f"Expected:  {true}")
        c += 1
        if c >= 5: break
    print(f"------------------------------------------------------------------------------")

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(running_loss.avg)
        else:
            scheduler.step()


def test_pipeline(model, loss_fn, loader, device):
    model.eval()

    t = time.time()
    running_string_metrics = StringMatchingMetrics()
    total_steps = len(loader)
    pbar = enumerate(loader)
    preds, trues = [], []

    for step, (imgs, image_labels) in pbar:
        imgs, image_labels = imgs.to(device, dtype=torch.float32).squeeze(0), image_labels
        curr_batch_size = imgs.size(0)

        # print(image_labels.shape, exam_label.shape)
        if config.MIXED_PRECISION_TRAIN:
            with torch.cuda.amp.autocast():
                image_preds = model(imgs)
                image_preds = image_preds.permute(1, 0, 2).view(1, -1, config.N_CLASSES)
                # print(image_preds.size(), loss)
                running_string_metrics.update(
                    y_pred=image_preds.detach().cpu(),
                    y_true=image_labels,
                    batch_size=curr_batch_size)

        else:
            image_preds = model(imgs)
            running_string_metrics.update(
                y_pred=image_preds.detach().cpu(),
                y_true=image_labels,
                batch_size=curr_batch_size)

            # print("Loss Update:", running_loss.avg)
            # print("Acc Update:", running_loss.avg)

        edit = running_string_metrics.avg_edit_distance
        wer = running_string_metrics.avg_wer
        mer = running_string_metrics.avg_mer
        wil = running_string_metrics.avg_wil
        pred, true = get_one_from_batch(image_preds, image_labels)
        preds.append(pred)
        trues.append(true)


        if ((config.LEARNING_VERBOSE and (step + 1) % config.VERBOSE_STEP == 0)) or ((step + 1) == total_steps) or ((step + 1) == 1):
            description = f'[{step + 1:>4d}/{total_steps:>4d}] ED: {edit:.4f} | WER: {wer:.4f} | Time: {(time.time() - t) / 60:.2f} m'
            print(description, flush=True)
            
        if config.DEBUG_MODE: break

    c = 0
    for pred, true in zip(preds, trues):
        print(f"------------------------------------------------------------------------------")
        print(f"Predicted: {pred}")
        print(f"Expected:  {true}")
        c += 1
        if c >= 5: break
    print(f"------------------------------------------------------------------------------")


def get_net(name):
    net = nets[name]()
    return net


def get_device(n):
    n = 0

    if not config.USE_GPU:
        print(f"Device:                      CPU")
        return torch.device('cpu')
    else:
        print(
            f"Device:                      GPU ({torch.cuda.get_device_name(0)})")
        return torch.device('cuda:' + str(n))
