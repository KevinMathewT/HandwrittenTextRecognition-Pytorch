import math

import torch
from torch import optim
from torch.optim.optimizer import Optimizer

from warmup_scheduler import GradualWarmupScheduler

from . import det_config


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                    for base_lr in self.base_lrs]


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * \
                        state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                            N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay']
                                     * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


def get_optimizer_and_scheduler(net, dataloader):
    m = 1
    print(f"World Size:                  {m}")

    m /= det_config.WARMUP_FACTOR
    print(f"Learning Rate Multiplier:    {m}")

    print(f"Start Learning Rate:         {det_config.LEARNING_RATE * m}")

    # Optimizers

    print(f"Optimizer:                   {det_config.OPTIMIZER}")
    if det_config.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            params=net.parameters(),
            lr=det_config.LEARNING_RATE * m,
            weight_decay=1e-5,
            amsgrad=False
        )
    elif det_config.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            net.parameters(), lr=det_config.LEARNING_RATE * m, weight_decay=0.001)
    elif det_config.OPTIMIZER == "RAdam":
        optimizer = RAdam(
            net.parameters(),
            lr=det_config.LEARNING_RATE * m
        )
    else:
        optimizer = optim.SGD(
            net.parameters(), lr=det_config.LEARNING_RATE * m)

    # Schedulers

    print(f"Scheduler:                   {det_config.SCHEDULER}")
    if det_config.SCHEDULER == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=0,
            factor=0.1,
            verbose=det_config.LEARNING_VERBOSE)
    elif det_config.SCHEDULER == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=5, eta_min=0)
    elif det_config.SCHEDULER == "OneCycleLR":
        steps_per_epoch = len(dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=1e-2,
            epochs=det_config.MAX_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.25,)
    elif det_config.SCHEDULER == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=(det_config.MAX_EPOCHS - det_config.WARMUP_EPOCHS), # // 2,
            T_mult=1,
            eta_min=1e-5,
            last_epoch=-1)
    elif det_config.SCHEDULER == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=2,
            gamma=0.1)
    else:
        scheduler = None

    print(f"Gradual Warmup:              {det_config.SCHEDULER_WARMUP}")
    if det_config.SCHEDULER_WARMUP:
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=det_config.WARMUP_FACTOR, total_epoch=det_config.WARMUP_EPOCHS, after_scheduler=scheduler)

    return optimizer, scheduler
