import torch
import torch.nn as nn
import torchvision

import timm
from timm.models.layers.conv2d_same import Conv2dSame
from vision_transformer_pytorch import VisionTransformer

from collections import OrderedDict

from .. import config


def getCNNBackbone():
    if config.CNN_BACKBONE == "ResNet18" or config.CNN_BACKBONE is None:
        resnet = timm.create_model(
            "resnet18", pretrained=config.BACKBONE_PRETRAINED)
        resnet.fc = nn.Identity()
        resnet.global_pool = nn.Identity()
        # self.resnet.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # self.resnet.layer3[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        resnet.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        resnet.layer4[0].downsample[0] = nn.Conv2d(
            256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        return resnet

    elif config.CNN_BACKBONE == "EfficientNetB0_NS":
        effnet_b0 = timm.create_model(
            "tf_efficientnet_b0_ns", pretrained=config.BACKBONE_PRETRAINED)
        effnet_b0.classifier = nn.Identity()
        effnet_b0.global_pool = nn.Identity()
        # effnet_b0.act2 = nn.Identity()
        effnet_b0.bn2 = nn.BatchNorm2d(
            512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        # effnet_b0.conv_head = nn.Identity()
        effnet_b0.conv_head = nn.Conv2d(
            320, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        effnet_b0.blocks[5][0].conv_dw = Conv2dSame(
            672, 672, kernel_size=(5, 5), stride=(1, 1), groups=672, bias=False)
        return effnet_b0

    elif config.CNN_BACKBONE == "EfficientNetB1_NS":
        effnet_b1 = timm.create_model(
            "tf_efficientnet_b1_ns", pretrained=config.BACKBONE_PRETRAINED)
        effnet_b1.classifier = nn.Identity()
        effnet_b1.global_pool = nn.Identity()
        # effnet_b1.act2 = nn.Identity()
        effnet_b1.bn2 = nn.BatchNorm2d(
            512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        # effnet_b1.conv_head = nn.Identity()
        effnet_b1.conv_head = nn.Conv2d(
            320, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        effnet_b1.blocks[5][0].conv_dw = Conv2dSame(
            672, 672, kernel_size=(5, 5), stride=(1, 1), groups=672, bias=False)
        return effnet_b1

class Image2TextRecurrentNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.RNN_INPUT_SIZE = config.RNN_INPUT_SIZE
        self.RNN_HIDDEN_SIZE = 256
        self.RNN_LAYERS = 2
        self.BIDIRECTIONAL = True
        self.RNN_DROPOUT = 0

        self.recurrent = nn.LSTM(
            input_size=self.RNN_INPUT_SIZE,
            hidden_size=self.RNN_HIDDEN_SIZE,
            num_layers=self.RNN_LAYERS,
            bidirectional=self.BIDIRECTIONAL,
            dropout=self.RNN_DROPOUT
        )
        self.reduce = nn.Linear(
            in_features=(1 + self.BIDIRECTIONAL) * self.RNN_HIDDEN_SIZE,
            out_features=config.N_CLASSES
        )

    def forward(self, x):
        output, _ = self.recurrent(x)
        t, b, h = output.size()
        output = output.view(t * b, h)

        reduced = self.reduce(output)
        log_probs = nn.LogSoftmax(dim=1)(reduced).view(t, b, config.N_CLASSES)

        return log_probs


class Image2TextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.RNN = Image2TextRecurrentNet()

        self.TIME_STEPS = config.TIME_STEPS
        self.cnn_backbone = getCNNBackbone()

    def forward(self, x):
        output = self.cnn_backbone(x)
        output = output.view(output.shape[0], output.shape[1], -1)
        output = output.permute(2, 0, 1).view(
            self.TIME_STEPS, -1, self.RNN.RNN_INPUT_SIZE)

        output = self.RNN(output)
        return output


nets = {
    "Image2TextNet": Image2TextNet,
}


# print(Image2TextNet())
# net1 = Image2TextConvNet().to(config.DEVICE)
# input1 = torch.rand((config.BATCH_SIZE, 1, config.IMAGE_H, config.IMAGE_W)).to(config.DEVICE)
# net2 = Image2TextNet().to(config.DEVICE)
# input2 = torch.rand((config.BATCH_SIZE, 1, config.IMAGE_H, config.IMAGE_W)).to(config.DEVICE)
# print(input1.size())
# print(net1(input1).size())
# print(net2(input2).size())
# print(torch.rand((1, 256, 8, 32)).size())
# print(nn.MaxPool2d((2, 2), (2, 1), (0, 1))(torch.rand((1, 256, 8, 32))).size())
