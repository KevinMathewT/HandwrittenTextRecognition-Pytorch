import torch
import torch.nn as nn
import torchvision

import timm
from vision_transformer_pytorch import VisionTransformer

from collections import OrderedDict

from .. import config

if config.USE_TPU:
    import torch_xla.core.xla_model as xm

class Image2TextConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = OrderedDict()

        self.CONV_CHANNELS   = [32, 64, 128, 128, 256]
        self.CONV_KERNEL     = [7, 5, 5, 3, 3, 3]
        self.CONV_STRIDE     = [1, 1, 1, 1, 1, 1]
        self.CONV_PADDING    = [3, 2, 2, 1, 1, 1]
        self.BATCH_NORM      = [1, 1, 1, 1, 1, 1] # 1 means we will have a Batch Normalization Layer
        self.LEAKY_RELU      = [0, 0, 0, 0, 0, 0]
        self.DROPOUT         = [0, 0, 0, 0, 0, 0] # 0 means we will not have any Dropout
        self.MAX_POOLING     = [ 
                [(2, 2), (2, 2)], 
                [(2, 2), (2, 2)], 
                [(1, 2), (1, 2)], 
                [(1, 2), (1, 2)], 
                [(1, 2), (1, 2)], 
                [], 
            ]
        self.NUM_LAYERS      = len(self.CONV_CHANNELS)

        # Convolution --> Batch Normalization --> ReLU / LeakyReLU --> Dropout --> MaxPooling
        for i in range(self.NUM_LAYERS):
            in_channels = config.C if i == 0 else self.CONV_CHANNELS[i - 1]
            out_channels = self.CONV_CHANNELS[i]

            layers["conv_%d" % (i)] = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.CONV_KERNEL[i],
                    stride=self.CONV_STRIDE[i],
                    padding=self.CONV_PADDING[i]
                )

            if self.BATCH_NORM[i]:
                layers["batch_norm_%d" % (i)] = nn.BatchNorm2d(num_features=out_channels)
            
            if self.LEAKY_RELU[i]:
                layers["leaky_relu_%d" % (i)] = nn.LeakyReLU(self.LEAKY_RELU)
            else:
                layers["relu_%d" % (i)] = nn.ReLU()

            if self.DROPOUT[i]:
                layers["dropout_%d" % (i)] = nn.Dropout(self.DROPOUT[i])
            
            if len(self.MAX_POOLING[i]) > 0:
                layers["max_pooling_%d" % (i)] = nn.MaxPool2d(*self.MAX_POOLING[i])

            
        # *[...] allows unpacking list elements as parameters to a function
        self.context_net = nn.Sequential(layers)

    def forward(self, x):
        return self.context_net(x)

class Image2TextRecurrentNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.RNN_INPUT_SIZE  = config.RNN_INPUT_SIZE
        self.RNN_HIDDEN_SIZE = 256
        self.RNN_LAYERS      = 2
        self.BIDIRECTIONAL   = True
        self.RNN_DROPOUT     = 0

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

        self.USE_RESNET      = True
        self.TIME_STEPS      = config.TIME_STEPS

        if self.USE_RESNET:
            self.resnet = timm.create_model("resnet18", pretrained=False)
            self.resnet.fc = nn.Identity()
            self.resnet.global_pool = nn.Identity()
            # self.resnet.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            # self.resnet.layer3[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.resnet.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.resnet.layer4[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        else:
            self.CNN = Image2TextConvNet()
 
    def forward(self, x):
        if self.USE_RESNET:
            output = self.resnet(x)
            output = output.view(output.shape[0], output.shape[1], -1)
            output = output.permute(2, 0, 1).view(self.TIME_STEPS, -1, self.RNN.RNN_INPUT_SIZE)
        else:
            output = self.CNN(x)
            output = output.squeeze(3)
            output = output.permute(2, 0, 1)

        output = self.RNN(output)
        return output


nets = {
    "Image2TextNet"         : Image2TextNet,
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