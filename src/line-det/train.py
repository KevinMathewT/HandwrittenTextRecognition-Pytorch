import os
import gc
import time
import warnings

import torch

from .dataset import get_loaders
from .engine import get_device, get_net, train_one_epoch, valid_one_epoch
from .loss import get_train_criterion, get_valid_criterion
from .optim import get_optimizer_and_scheduler
from . import det_config
from .. import config
from .. import utils

warnings.filterwarnings("ignore")


def run_fold(fold):
    utils.create_dirs(net=det_config.NET)
    print(f"------------------------------------------------------------------------------")
    print(f"Training Model:              {det_config.NET}")
    print(f"Maximum Epochs:              {det_config.MAX_EPOCHS}")
    print(f"Training Fold:               {fold}")
    print(f"Image Dimensions:            {det_config.H}x{det_config.W}")
    print(f"Training Batch Size:         {det_config.TRAIN_BATCH_SIZE}")
    print(f"Validation Batch Size:       {det_config.VALID_BATCH_SIZE}")

    global net
    train_loader, valid_loader = get_loaders(fold)
    device = get_device(n=0)
    net = net.to(device)
    loss_tr = get_train_criterion(device=device)
    loss_fn = get_valid_criterion(device=device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        net=net, dataloader=train_loader)

    gc.collect()

    for epoch in range(det_config.MAX_EPOCHS):
        epoch_start = time.time()
        if det_config.DO_FREEZE_BATCH_NORM and epoch < det_config.FREEZE_BN_EPOCHS:
            utils.freeze_batchnorm_stats(net)

        print(f"------------------------------------------------------------------------------")
        train_one_epoch(fold, epoch, net, loss_tr, optimizer, train_loader, device,
                        scheduler=scheduler, schd_batch_update=det_config.SCHEDULER_BATCH_STEP)

        valid_one_epoch(fold, epoch, net, loss_fn, valid_loader, device)

        print(f'[{fold}/{det_config.FOLDS - 1}][{epoch:>2d}/{det_config.MAX_EPOCHS - 1:>2d}] Time Taken for Epoch {epoch}: {time.time() - epoch_start} seconds |')

        torch.save(net.state_dict(), os.path.join(config.WEIGHTS_PATH,
                                                  f'{det_config.NET}/{det_config.NET}_fold_{fold}_{epoch}.bin'))
        print(f'[{fold}/{det_config.FOLDS - 1}][{epoch:>2d}/{det_config.MAX_EPOCHS - 1:>2d}] Model saved at {os.path.join(config.WEIGHTS_PATH, f"{det_config.NET}/{det_config.NET}_fold_{fold}_{epoch}.bin")}')

    # torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
    del net, optimizer, train_loader, valid_loader, scheduler
    torch.cuda.empty_cache()
    print(f"------------------------------------------------------------------------------")


def train():
    global net
    torch.cuda.empty_cache()
    for fold in [0]:
        net = get_net(name=det_config.NET)
        run_fold(fold)


if __name__ == "__main__":
    train()
