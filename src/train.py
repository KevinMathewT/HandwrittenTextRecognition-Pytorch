import gc
import time
from joblib import Parallel, delayed

import torch
import torch.nn as nn

from .dataset import get_loaders
from .optim import get_optimizer_and_scheduler
from .engine import get_device, get_net, train_one_epoch, valid_one_epoch
from . import config
from .utils import *
from .loss import get_train_criterion, get_valid_criterion

import warnings
warnings.filterwarnings("ignore")


def run_fold(fold):
    create_dirs()
    print(f"------------------------------------------------------------------------------")
    print(f"Training Model:              {config.NET}")
    print(f"Training Fold:               {fold}")
    print(f"Image Dimensions:            {config.H}x{config.W}")
    print(f"CNN Backbone:                {config.CNN_BACKBONE}")
    print(f"Mixed Precision Training:    {config.MIXED_PRECISION_TRAIN}")
    print(f"Training Batch Size:         {config.TRAIN_BATCH_SIZE}")
    print(f"Validation Batch Size:       {config.VALID_BATCH_SIZE}")
    print(f"Accumulate Iteration:        {config.ACCUMULATE_ITERATION}")
    print(f"CTC Decoder:                 {config.DECODER}")

    global net
    train_loader, valid_loader = get_loaders(fold)
    device = get_device(n=fold+1)
    net = net.to(device)
    scaler = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION_TRAIN else None
    loss_tr = get_train_criterion(device=device)
    loss_fn = get_valid_criterion(device=device)
    optimizer, scheduler = get_optimizer_and_scheduler(
        net=net, dataloader=train_loader)

    gc.collect()

    for epoch in range(config.MAX_EPOCHS):
        epoch_start = time.time()
        if config.DO_FREEZE_BATCH_NORM and epoch < config.FREEZE_BN_EPOCHS:
            freeze_batchnorm_stats(net)

        train_one_epoch(fold, epoch, net, loss_tr, optimizer, train_loader, device,
                        scaler=scaler, scheduler=scheduler, schd_batch_update=config.SCHEDULER_BATCH_STEP)

        valid_one_epoch(fold, epoch, net, loss_fn, valid_loader,
                        device, scheduler=None, schd_loss_update=False)

        print(f'[{fold}/{config.FOLDS - 1}][{epoch:>2d}/{config.MAX_EPOCHS - 1:>2d}] Time Taken for Epoch {epoch}: {time.time() - epoch_start} seconds |')
        print(f"------------------------------------------------------------------------------")                                                                      

        torch.save(net.state_dict(), os.path.join(config.WEIGHTS_PATH,
                                                  f'{config.NET}/{config.NET}_fold_{fold}_{epoch}.bin'))

    #torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))
    del net, optimizer, train_loader, valid_loader, scheduler
    torch.cuda.empty_cache()
    print(f"------------------------------------------------------------------------------")


def train():
    global net
    torch.cuda.empty_cache()
    for fold in [0]:
        net = get_net(name=config.NET)
        run_fold(fold)


if __name__ == "__main__":
    train()
