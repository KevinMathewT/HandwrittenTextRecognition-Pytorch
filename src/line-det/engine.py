import time

import torch

from ..utils import AverageLossMeter
from .model import nets
from . import det_config


def train_one_epoch(fold, epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    t = time.time()
    model.train()
    loss_fn.train()
    summary_loss = AverageLossMeter()
    total_steps = len(train_loader)

    for step, (images, targets) in enumerate(train_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        curr_batch_size = len(images)

        output = model(images)

        loss_dict = loss_fn(output, targets)
        weight_dict = loss_fn.weight_dict

        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        if scheduler is not None and schd_batch_update:
            scheduler.step(epoch + (step / total_steps))

        summary_loss.update(curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)

        loss = summary_loss.avg
        if (det_config.LEARNING_VERBOSE and (step + 1) % det_config.VERBOSE_STEP == 0) or ((step + 1) == total_steps) or ((step + 1) == 1):
            description = f'[{fold}/{det_config.FOLDS - 1}][{epoch:>2d}/{det_config.MAX_EPOCHS - 1:>2d}][{step + 1:>4d}/{total_steps:>4d}] Loss: {loss:.4f} | Time: {(time.time() - t) / 60:.2f} m'
            print(description, flush=True)

        break


def valid_one_epoch(fold, epoch, model, loss_fn, valid_loader, device):
    t = time.time()
    model.eval()
    loss_fn.eval()
    summary_loss = AverageLossMeter()
    total_steps = len(valid_loader)

    with torch.no_grad():
        for step, (images, targets) in enumerate(valid_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            curr_batch_size = len(images)

            output = model(images)

            loss_dict = loss_fn(output, targets)
            weight_dict = loss_fn.weight_dict

            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            summary_loss.update(loss.item(), curr_batch_size)

            loss = summary_loss.avg
            if (det_config.LEARNING_VERBOSE and (step + 1) % det_config.VERBOSE_STEP == 0) or ((step + 1) == total_steps) or ((step + 1) == 1):
                description = f'[{fold}/{det_config.FOLDS - 1}][{epoch:>2d}/{det_config.MAX_EPOCHS - 1:>2d}][{step + 1:>4d}/{total_steps:>4d}] Loss: {loss:.4f} | Time: {(time.time() - t) / 60:.2f} m'
                print(description, flush=True)

            break

def get_net(name):
    net = nets[name]()
    return net


def get_device(n):
    if not det_config.USE_GPU:
        print(f"Device:                      CPU")
        return torch.device('cpu')
    else:
        print(f"Device:                      GPU ({torch.cuda.get_device_name(n)})")
        return torch.device('cuda:' + str(n))
