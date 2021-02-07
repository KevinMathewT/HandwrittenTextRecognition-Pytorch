import torch
import torch.nn as nn

from . import config
from .utils import stringToClasses

if config.USE_TPU:
    import torch_xla.core.xla_model as xm


class MyCTCLoss(nn.Module):
    def __init__(self, device):
        super(MyCTCLoss, self).__init__()
        self.device = device

    def forward(self, outputs, targets):
        outputs = outputs.to(self.device)
        output_lengths = torch.full(
            size=(outputs.size()[1],),
            fill_value=config.TIME_STEPS,
            dtype=torch.long
        ).to(self.device)

        # +------------------------------

        target_val = torch.ones((1), dtype=torch.long)
        for target in targets:
            target_val = torch.cat(
                (target_val, stringToClasses(target)), 0)

        # +------------------------------ ABOVE OR BELOW------------------------------

        # targets = [line + (" " * (config.TIME_STEPS - len(line))) for line in targets]
        # target_val = torch.ones((1, config.TIME_STEPS), dtype=torch.long)
        # for target in targets:
        #     target_val = torch.cat((target_val, utils.stringToClasses(target).unsqueeze(0)), 0)

        # +------------------------------

        target_val = target_val[1:].to(self.device)
        target_lengths = torch.tensor(
            [len(target) for target in targets],
            dtype=torch.long
        ).to(self.device)

        # print("Outputs:")
        # print(outputs)
        # print("Outputs Lengths:")
        # print(output_lengths)
        # print("Targets:")
        # print(target_val)
        # print("Target Lengths:")
        # print(target_lengths)

        loss = nn.CTCLoss(zero_infinity=True)(
            outputs, target_val, output_lengths, target_lengths)
        return loss


def get_train_criterion(device):
    print_fn = print if not config.USE_TPU else xm.master_print
    print_fn(f"Training Criterion:          {config.TRAIN_CRITERION}")

    losses = {
        "CTCLoss": MyCTCLoss(device=device).to(device),
    }

    return losses[config.VALID_CRITERION]

    # if config.TRAIN_CRITERION == "BiTemperedLogisticLoss":
    #     return bi_tempered_logistic_loss
    # elif config.TRAIN_CRITERION == "SoftmaxCrossEntropy_OHL":  # For One Hot Labels
    #     return MyCrossEntropyLoss().to(device)
    # elif config.TRAIN_CRITERION == "SoftmaxCrossEntropy":  # For One Hot Labels
    #     return nn.CrossEntropyLoss().to(device)
    # elif config.TRAIN_CRITERION == "FocalCosineLoss":
    #     return FocalCosineLoss(device=device).to(device)
    # elif config.TRAIN_CRITERION == "LabelSmoothingCrossEntropy":
    #     return LabelSmoothingCrossEntropy(smoothing=0.2).to(device)
    # elif config.TRAIN_CRITERION == "SmoothCrossEntropyLoss":
    #     return SmoothCrossEntropyLoss(smoothing=0.1).to(device)
    # elif config.TRAIN_CRITERION == "TaylorCrossEntropyLoss":
    #     return TaylorCrossEntropyLoss().to(device)
    # elif config.TRAIN_CRITERION == "RandomChoice":
    #     return RandomLoss(device).to(device)
    # elif config.TRAIN_CRITERION == "TaylorCrossEntropyLossWithLabelSmoothing":
    #     return TaylorCrossEntropyLossWithLabelSmoothing().to(device)
    # elif config.TRAIN_CRITERION == "CTCLoss":
    #     return MyCTCLoss(device=device).to(device)
    # else:
    #     return nn.CrossEntropyLoss().to(device)


def get_valid_criterion(device):
    print_fn = print if not config.USE_TPU else xm.master_print
    print_fn(f"Validation Criterion:        {config.VALID_CRITERION}")

    losses = {
        "CTCLoss": MyCTCLoss(device=device).to(device),
    }

    return losses[config.VALID_CRITERION]

    # if config.VALID_CRITERION == "BiTemperedLogisticLoss":
    #     return bi_tempered_logistic_loss
    # elif config.VALID_CRITERION == "SoftmaxCrossEntropy_OHL":  # For One Hot Labels
    #     return MyCrossEntropyLoss().to(device)
    # elif config.VALID_CRITERION == "SoftmaxCrossEntropy":  # For One Hot Labels
    #     return nn.CrossEntropyLoss().to(device)
    # elif config.VALID_CRITERION == "FocalCosineLoss":
    #     return FocalCosineLoss(device=device).to(device)
    # elif config.TRAIN_CRITERION == "LabelSmoothingCrossEntropy":
    #     return LabelSmoothingCrossEntropy().to(device)
    # elif config.VALID_CRITERION == "SmoothCrossEntropyLoss":
    #     return SmoothCrossEntropyLoss(smoothing=0.1).to(device)
    # elif config.VALID_CRITERION == "TaylorCrossEntropyLoss":
    #     return TaylorCrossEntropyLoss().to(device)
    # elif config.VALID_CRITERION == "RandomChoice":
    #     return RandomLoss(device).to(device)
    # elif config.VALID_CRITERION == "TaylorCrossEntropyLossWithLabelSmoothing":
    #     return TaylorCrossEntropyLossWithLabelSmoothing().to(device)
    # elif config.VALID_CRITERION == "CTCLoss":
    #     return MyCTCLoss(device=device).to(device)
    # else:
    #     return nn.CrossEntropyLoss().to(device)
