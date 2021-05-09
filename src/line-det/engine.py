import time
import numpy as np

import torch

from ..utils import AverageLossMeter
from .model import nets
from . import det_config
from .det_utils import calculate_image_precision


def train_one_epoch(fold, epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None,
                    schd_batch_update=False):
    t = time.time()
    model.train()
    loss_fn.train()
    summary_loss = AverageLossMeter()
    total_steps = len(train_loader)

    for step, (images, targets, ids) in enumerate(train_loader):
        try:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            curr_batch_size = len(images)

            if det_config.WEIGHTED_LOSS:
                output = model(images)
                loss_dict = loss_fn(output, targets)
                weight_dict = loss_fn.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            else:
                loss_dict = model(images, targets)
                print(loss_dict)
                loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None and schd_batch_update:
                scheduler.step(epoch + (step / total_steps))

            summary_loss.update(curr_batch_avg_loss=loss.item(), batch_size=curr_batch_size)

            loss = summary_loss.avg
            if (det_config.LEARNING_VERBOSE and (step + 1) % det_config.VERBOSE_STEP == 0) or (
                    (step + 1) == total_steps) or ((step + 1) == 1):
                description = f'[{fold}/{det_config.FOLDS - 1}][{epoch:>2d}/{det_config.MAX_EPOCHS - 1:>2d}][{step + 1:>4d}/{total_steps:>4d}] Loss: {loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.8f} | Time: {(time.time() - t) / 60:.2f} m'
                print(description, flush=True)

            break

        except Exception as e:
            print(f"{e}\nError for ids: {ids}")


def valid_one_epoch(fold, epoch, model, loss_fn, valid_loader, device):
    t = time.time()
    model.eval()
    loss_fn.eval()
    summary_loss = AverageLossMeter()
    summary_iou = AverageLossMeter()
    total_steps = len(valid_loader)
    validation_image_precisions = []
    iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]

    with torch.no_grad():
        try:
            for step, (images, targets, ids) in enumerate(valid_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                curr_batch_size = len(images)

                predictions = model(images)
                # print(predictions)

                for i, image in enumerate(images):
                    boxes = predictions[i]['boxes'].data.cpu().numpy() / (det_config.H - 1)
                    scores = predictions[i]['scores'].data.cpu().numpy()
                    labels = np.ones(predictions[i]['scores'].shape[0])

                    # boxes = np.array(boxes).astype(np.int32).clip(min=0, max=1023)
                    preds = boxes # outputs[i]['boxes'].data.cpu().numpy()
                    # scores = outputs[i]['scores'].data.cpu().numpy()
                    preds_sorted_idx = np.argsort(scores)[::-1]
                    preds_sorted = preds[preds_sorted_idx]
                    gt_boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
                    image_precision = calculate_image_precision(preds_sorted,
                                                                gt_boxes,
                                                                thresholds=iou_thresholds,
                                                                form='coco')
                    summary_iou.update(image_precision)

                if det_config.WEIGHTED_LOSS:
                    output = model(images)
                    loss_dict = loss_fn(output, targets)
                    weight_dict = loss_fn.weight_dict
                    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                else:
                    loss_dict = model(images, targets)
                    # print(loss_dict)
                    loss = sum(loss for loss in loss_dict.values())

                summary_loss.update(loss.item(), curr_batch_size)

                loss = summary_loss.avg
                iou = summary_iou.avg
                if (det_config.LEARNING_VERBOSE and (step + 1) % det_config.VERBOSE_STEP == 0) or (
                        (step + 1) == total_steps) or ((step + 1) == 1):
                    description = f'[{fold}/{det_config.FOLDS - 1}][{epoch:>2d}/{det_config.MAX_EPOCHS - 1:>2d}][{step + 1:>4d}/{total_steps:>4d}] Loss: {loss:.4f} | IOU: {iou} | Time: {(time.time() - t) / 60:.2f} m'
                    print(description, flush=True)

                break
        except Exception as e:
            print(f"{e}\nError for ids: {ids}")


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
