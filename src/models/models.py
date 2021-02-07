import torch
import torch.nn as nn
import torchvision

import timm
from vision_transformer_pytorch import VisionTransformer

from collections import OrderedDict

from .. import config

if config.USE_TPU:
    import torch_xla.core.xla_model as xm


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class BinaryHead(nn.Module):
    def __init__(self, num_class=config.N_CLASSES, emb_size=2048, s=16.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit


class SEResNeXt50_32x4d_BH(nn.Module):
    name = "SEResNeXt50_32x4d_BH"

    def __init__(self, pretrained=config.PRETRAINED):
        super().__init__()
        self.model_arch = "seresnext50_32x4d"
        self.net = nn.Sequential(*list(
            timm.create_model(self.model_arch, pretrained=pretrained).children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(config.N_CLASSES, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        img_feature = self.net(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        # fea = self.dropout(fea)
        output = self.binary_head(fea)

        return output


class ResNeXt50_32x4d_BH(nn.Module):
    name = "ResNeXt50_32x4d_BH"

    def __init__(self, pretrained=config.PRETRAINED):
        super().__init__()
        self.model_arch = "resnext50_32x4d"
        self.model = timm.create_model(self.model_arch, pretrained=pretrained)
        model_list = list(self.model.children())
        model_list[-1] = nn.Identity()
        model_list[-2] = nn.Identity()
        self.net = nn.Sequential(*model_list)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(config.N_CLASSES, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=2048, out_features=config.N_CLASSES)

    def forward(self, x):
        x = self.net(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fea_bn(x)
        # fea = self.dropout(fea)
        x = self.binary_head(x)
        # x = self.fc(x)

        return x


class ViTBase16_BH(nn.Module):
    name = "ViTBase16_BH"

    def __init__(self, pretrained=config.PRETRAINED):
        super().__init__()
        self.net = timm.create_model(
            "vit_base_patch16_384", pretrained=pretrained)
        self.net.norm = nn.Identity()
        self.net.head = nn.Identity()
        self.fea_bn = nn.BatchNorm1d(768)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(config.N_CLASSES, emb_size=768, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.net(x)
        x = self.fea_bn(x)
        # fea = self.dropout(fea)
        x = self.binary_head(x)
        return x


class ViTBase16(nn.Module):
    name = "ViTBase16"

    def __init__(self, pretrained=config.PRETRAINED):
        super().__init__()
        # self.model_arch = 'ViT-B_16'
        # self.net = VisionTransformer.from_pretrained(
        #     self.model_arch, num_classes=5) if pretrained else VisionTransformer.from_name(self.model_arch, num_classes=5)
        # print(self.model)

        self.model_arch = 'vit_base_patch16_384'
        self.net = timm.create_model(self.model_arch, pretrained=pretrained)
        n_features = self.net.head.in_features
        self.net.head = nn.Linear(n_features, config.N_CLASSES)

    def forward(self, x):
        x = self.net(x)
        return x


class ViTLarge16(nn.Module):
    name = "ViTLarge16"

    def __init__(self, pretrained=config.PRETRAINED):
        super().__init__()
        # self.model_arch = 'ViT-B_16'
        # self.net = VisionTransformer.from_pretrained(
        #     self.model_arch, num_classes=5) if pretrained else VisionTransformer.from_name(self.model_arch, num_classes=5)
        # print(self.model)

        self.model_arch = 'vit_large_patch16_384'
        self.net = timm.create_model(self.model_arch, pretrained=pretrained)
        n_features = self.net.head.in_features
        self.net.head = nn.Linear(n_features, config.N_CLASSES)

    def forward(self, x):
        x = self.net(x)
        return x


class EfficientNetB4(nn.Module):
    name = "EfficientNetB4"

    def __init__(self, pretrained=config.PRETRAINED):
        super().__init__()
        self.model_arch = 'tf_efficientnet_b4_ns'
        self.net = timm.create_model(self.model_arch, pretrained=pretrained)
        n_features = self.net.classifier.in_features
        self.net.classifier = nn.Linear(n_features, config.N_CLASSES)

    def forward(self, x):
        x = self.net(x)
        return x


class EfficientNetB3(nn.Module):
    name = "EfficientNetB3"

    def __init__(self, pretrained=config.PRETRAINED):
        super().__init__()
        self.model_arch = 'tf_efficientnet_b3_ns'
        self.net = timm.create_model(self.model_arch, pretrained=pretrained)
        n_features = self.net.classifier.in_features
        self.net.classifier = nn.Linear(n_features, config.N_CLASSES)

    def forward(self, x):
        x = self.net(x)
        return x


class GeneralizedCassavaClassifier(nn.Module):
    def __init__(self, model_arch, n_class=config.N_CLASSES, pretrained=config.PRETRAINED):
        super().__init__()
        self.name = model_arch
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        model_list = list(self.model.children())
        model_list[-1] = nn.Linear(
            in_features=model_list[-1].in_features,
            out_features=n_class,
            bias=True
        )
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        x = self.model(x)
        return x

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

        self.RNN_INPUT_SIZE  = 256
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

        self.USE_RESNET      = False
        self.TIME_STEPS      = 32

        if self.USE_RESNET:
            self.resnet = torchvision.models.resnet50(pretrained=config.PRETRAINED)
            self.resnet_input = nn.Conv2d(
                        in_channels=1,
                        out_channels=3,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.fc1 = nn.Linear(1000, self.TIME_STEPS * self.RNN.RNN_INPUT_SIZE)
        else:
            self.CNN = Image2TextConvNet()
 
    def forward(self, x):
        if self.USE_RESNET:
            output = self.fc1(nn.ReLU()(self.resnet(self.resnet_input(x)))).view(self.TIME_STEPS, -1, self.RNN.RNN_INPUT_SIZE)
        else:
            output = self.CNN(x)
            output = output.squeeze(3)
            output = output.permute(2, 0, 1)

        output = self.RNN(output)
        return output


nets = {
    "SEResNeXt50_32x4d_BH"  : SEResNeXt50_32x4d_BH,
    "ViTBase16_BH"          : ViTBase16_BH,
    "ResNeXt50_32x4d_BH"    : ResNeXt50_32x4d_BH,
    "ViTBase16"             : ViTBase16,
    "ViTLarge16"            : ViTLarge16,
    "EfficientNetB4"        : EfficientNetB4,
    "EfficientNetB3"        : EfficientNetB3,
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