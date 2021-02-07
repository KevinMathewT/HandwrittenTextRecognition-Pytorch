import math
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomBrightness, Transpose, ToGray, Rotate,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, RandomContrast, GaussianBlur,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations import LongestMaxSize
import cv2
from skimage import transform
from torch.nn.modules.utils import _pair, _quadruple

from . import config
from .utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class TransposeImage(object):
    def __init__(self):
        return

    def __call__(self, image, force_apply):
        image = torch.tensor(image.copy()).permute(2, 0, 1)
        image = torch.transpose(image, 1, 2)
        image = image.permute(1, 2, 0).numpy()
        return {
            "image": image
        }


class ToGrayscale(object):
    def __init__(self):
        return

    def rgb2gray(self, rgb):
        r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def __call__(self, image, force_apply):
        image = torch.tensor(image.copy()).permute(2, 0, 1)
        image = self.rgb2gray(image).unsqueeze(0)
        image = image.permute(1, 2, 0).numpy()
        return {
            "image": image
        }


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, force_apply):
        image = torch.tensor(image.copy()).permute(2, 0, 1)

        h, w = image.shape[:2]
        if (h / w) > (self.output_size[0] / self.output_size[1]):
            req_w = (h * self.output_size[1]) / self.output_size[0]
            image = F.pad(image, (0, int(req_w) - w, 0, 0),
                          value=239, mode="constant")
        elif (h / w) < (self.output_size[0] / self.output_size[1]):
            req_h = (w * self.output_size[0]) / self.output_size[1]
            image = F.pad(image, (0, 0, int((req_h - h) // 2),
                                  int((req_h - h) // 2)), value=239, mode="constant")

        # print(image.shape)

        new_h, new_w = self.output_size
        image = F.interpolate(torch.tensor(image).unsqueeze(0), size=(new_h, new_w)).squeeze(0)

        image = image.view(-1, new_h, new_w)
        image = image.permute(1, 2, 0).numpy()
        return {
            "image": image
        }


class GreyscaleToBlackAndWhite(object):
    def __init__(self):
        self.threshold = config.THRESHOLD

    def __call__(self, image, force_apply):
        image = torch.tensor(image.copy()).permute(2, 0, 1)
        image = (image <= self.threshold).type('torch.FloatTensor')
        image = image.permute(1, 2, 0).numpy()
        return {
            "image": image
        }


class GaussianFiltering(object):
    def __init__(self, channels=1, kernel_size=7, sigma=3):
        # Set these to whatever you want for your gaussian filter
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        self.x_cord = torch.arange(self.kernel_size)
        self.x_grid = self.x_cord.repeat(self.kernel_size).view(
            self.kernel_size, self.kernel_size)
        self.y_grid = self.x_grid.t()
        self.xy_grid = torch.stack([self.x_grid, self.y_grid], dim=-1)

        self.mean = (self.kernel_size - 1) / 2.
        self.variance = self.sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        self.gaussian_kernel = (1./(2. * math.pi * self.variance)) *\
            torch.exp(
            -torch.sum((self.xy_grid - self.mean)**2., dim=-1) /
            (2*self.variance)
        )
        # Make sure sum of values in gaussian kernel equals 1.
        self.gaussian_kernel = self.gaussian_kernel / \
            torch.sum(self.gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        self.gaussian_kernel = self.gaussian_kernel.view(
            1, 1, self.kernel_size, self.kernel_size)
        self.gaussian_kernel = self.gaussian_kernel.repeat(
            self.channels, 1, 1, 1)

        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                         kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=self.channels, bias=False)

        self.gaussian_filter.weight.data = self.gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def __call__(self, image, force_apply):
        image = torch.tensor(image.copy()).permute(2, 0, 1)
        image = self.gaussian_filter(image.unsqueeze(0).float()).squeeze(0)
        image = image.permute(1, 2, 0).numpy()
        return {
            "image": image
        }


class AverageFiltering(object):
    def __init__(self, channels=1, kernel_size=5):
        # Set these to whatever you want for your gaussian filter
        self.kernel_size = kernel_size
        self.channels = channels

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        self.averaging_kernel = torch.ones(
            (self.kernel_size, self.kernel_size))
        self.averaging_kernel = self.averaging_kernel / \
            torch.sum(self.averaging_kernel)

        # Reshape to 2d depthwise convolutional weight
        self.averaging_kernel = self.averaging_kernel.view(
            1, 1, self.kernel_size, self.kernel_size)
        self.averaging_kernel = self.averaging_kernel.repeat(
            self.channels, 1, 1, 1)

        self.averaging_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                          kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=self.channels, bias=False)

        self.averaging_filter.weight.data = self.averaging_kernel
        self.averaging_filter.weight.requires_grad = False

    def __call__(self, image, force_apply):
        image = torch.tensor(image.copy()).permute(2, 0, 1)
        image = self.averaging_filter(image.unsqueeze(0)).squeeze(0)
        image = image.permute(1, 2, 0).numpy()
        return {
            "image": image
        }


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x, force_apply):
        x = torch.tensor(x.copy()).permute(2, 0, 1)
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x.unsqueeze(0), self._padding(
            x.unsqueeze(0)), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(
            3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        x = x.squeeze(0)
        x = x
        x = x.permute(1, 2, 0).numpy()
        return {
            "image": x
        }


class NumpyToTensor(object):
    def __init__(self):
        return

    def __call__(self, image):
        return torch.Tensor(image.float())

# tensor([237.1252, 237.1252, 237.1252], device='cuda:0')
# tensor([42.7399, 42.7399, 42.7399], device='cuda:0')


def get_train_transforms():
    return Compose([
        Rescale((config.H, config.W)),
        TransposeImage(),
        # ShiftScaleRotate(p=1.),
        Rotate(p=1., border_mode=cv2.BORDER_CONSTANT, value=[255., 255., 255.], limit=10.),
        # Normalize(mean=[237.1252, 237.1252, 237.1252], std=[42.7399, 42.7399, 42.7399], p=1.0),
        ToGrayscale(),

        # GaussianFiltering(channels=1, kernel_size=5, sigma=1),
        GreyscaleToBlackAndWhite(),
        ToTensorV2(p=1.0),
    ])

    return Compose([
        RandomResizedCrop(config.H, config.W),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2,
                           sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[
                  0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms():
    return Compose([
        Resize(config.H, config.W),
        Normalize(mean=[0.485, 0.456, 0.406], std=[
                  0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_inference_transforms():
    return Compose([
        RandomResizedCrop(config.H, config.W),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2,
                           sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[
                  0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)
