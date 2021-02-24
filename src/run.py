from .dataset import get_loaders, HandWritingLinesDataset
from .transforms import get_train_transforms
from . import config
from tqdm import tqdm

import pandas as pd

import torch
import torch.nn as nn
from torchvision.utils import save_image

import timm

if __name__ == '__main__':
#     fold = 0
#     # train_folds = pd.read_csv(config.TRAIN_FOLDS)
#     # train = train_folds[train_folds.fold != fold]
#     # valid = train_folds[train_folds.fold == fold]
#     # train_dataset = HandWritingLinesDataset(train, transforms=get_train_transforms())
    
#     # train_loader, valid_loader          = get_loaders(fold)

#     # nimages = 0
#     # mean = 0.
#     # std = 0.
#     # for batch, _ in tqdm(train_loader, total=len(train_loader)):
#     #     batch = torch.Tensor.float(batch.view(batch.size(0), batch.size(1), -1).cuda())
#     #     nimages += batch.size(0)
#     #     mean += batch.mean(2).sum(0) 
#     #     std += batch.std(2).sum(0)

#     # mean /= nimages
#     # std /= nimages

#     # print(mean)
#     # print(std)

#     # for a, b in train_loader:
#     #     print(a.size())
#     #     print(len(b))
#     #     break

#     # print(timm.list_models('*res*'))

#     input = torch.ones((4, 3, config.H, config.W))
#     # net = timm.create_model("resnet18", pretrained=False)
#     # # print(net)
#     # net.fc = nn.Identity()
#     # net.global_pool = nn.Identity()
#     # # net.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     # # net.layer3[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     # net.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     # net.layer4[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     # # # net.bn2 = nn.Identity()
#     # # # net.conv_head = nn.Identity()
#     # # # net.blocks[6] = nn.Identity()
#     # out = net(input)
#     # out = out.view(out.shape[0], out.shape[1], -1)
#     # out = out.permute(2, 0, 1)
#     # print(out.size())

    
#     resnet = timm.create_model("resnet18", pretrained=False)
#     resnet.fc = nn.Identity()
#     resnet.global_pool = nn.Identity()
#     # self.resnet.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     # self.resnet.layer3[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#     resnet.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     resnet.layer4[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    
#     output = resnet(input)
#     output = output.view(output.shape[0], output.shape[1], -1)
#     output = output.permute(2, 0, 1).view(config.TIME_STEPS, -1, config.RNN_INPUT_SIZE)

#     print(output.size())

#     # a = torch.Tensor([i for i in range(64)])
#     # a = a.view(2, 2, 4, 4)
#     # print(a)
#     # a = a.view(2, 2, 16)
#     # print(a)
#     # a = a.view(2, 4, 8)
#     # print(a)
#     # a = a.permute(1, 0, 2).contiguous()
#     # print(a)
#     # print(a.view(2, -1))

    input = torch.ones((4, 3, config.H, config.W))
    resnet = timm.create_model("resnet18", pretrained=False)
    resnet.fc = nn.Identity()
    resnet.global_pool = nn.Identity()
    # resnet.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # resnet.layer3[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    resnet.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    resnet.layer4[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # torch.Size([4, 512, 64, 4])

    # print(timm.list_models("*eff*"))

    effnet_b0 = timm.create_model("tf_efficientnet_b0_ns", pretrained=False)
    effnet_b0.classifier = nn.Identity()
    effnet_b0.global_pool = nn.Identity()
    effnet_b0.act2 = nn.Identity()
    effnet_b0.bn2 = nn.Identity()
    # effnet_b0.conv_head = nn.Identity()
    effnet_b0.conv_head = nn.Conv2d(192, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    effnet_b0.blocks[6] = nn.Identity()
    effnet_b0.blocks[5][0].conv_dw = nn.Identity()
    # effnet_b0.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # effnet_b0.layer3[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # effnet_b0.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # effnet_b0.layer4[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # torch.Size([4, 512, 64, 4])
    effnet_b0_total_params = sum(p.numel() for p in effnet_b0.parameters())
    effnet_b0_trainable_total_params = sum(p.numel() for p in effnet_b0.parameters() if p.requires_grad)
    resnet_total_params = sum(p.numel() for p in resnet.parameters())
    resnet_trainable_total_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    print(f"EfficientNetB0 Total Params: {effnet_b0_total_params}")
    print(f"EfficientNetB0 Total Trainable Params: {effnet_b0_trainable_total_params}")
    print(f"ResNet18 Total Params: {resnet_total_params}")
    print(f"ResNet18 Total Trainable Params: {resnet_trainable_total_params}")

    output = effnet_b0(input)
    print(input.size())
    print(output.size())


'''
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act1): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): ReLU(inplace=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): ReLU(inplace=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): ReLU(inplace=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): ReLU(inplace=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): ReLU(inplace=True)
    )
  )
  (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=True)
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
)
'''