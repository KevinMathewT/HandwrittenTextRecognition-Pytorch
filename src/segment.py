import cv2
import math
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import warnings
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')
# %matplotlib inline

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import argrelmin


def createKernel(kernelSize, sigma, theta):
    "create anisotropic filter kernel according to given parameters"
    assert kernelSize % 2  # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
            xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
            yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel


def applySummFunctin(img):
    res = np.sum(img, axis=0)  # summ elements in columns
    return res


def normalize(img):
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img
    return img


def smooth(x, window_len=11, window='hanning'):
    #     if x.ndim != 1:
    #         raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y

def crop_text_to_lines(text, blanks):
    x1 = 0
    y = 0
    lines = []
    for i, blank in enumerate(blanks):
        x2 = blank
        print("x1=", x1, ", x2=", x2, ", Diff= ", x2-x1)
        line = text[:, x1:x2]
        lines.append(line)
        x1 = blank
    return lines

def lineSegment(img):
    img1 = img
    print(img1.ndim)
    print(img1.shape)
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    print(img2.shape)
    print(type(img2))
    img3 = np.transpose(img2)
    img = np.arange(16).reshape((4, 4))

    kernelSize = 9
    sigma = 4
    theta = 1.5
    #25, 0.8, 3.5

    imgFiltered1 = cv2.filter2D(
        img3, -1, createKernel(kernelSize, sigma, theta), borderType=cv2.BORDER_REPLICATE)
    img4 = normalize(imgFiltered1)

    (m, s) = cv2.meanStdDev(imgFiltered1)
    print(m[0][0])

    summ = applySummFunctin(img4)
    print(summ.ndim)
    print(summ.shape)

    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    smoothed = smooth(summ, 35)
    # plt.plot(smoothed)
    # plt.show()

    mins = argrelmin(smoothed, order=2)
    arr_mins = np.array(mins)

    found_lines = crop_text_to_lines(img3, arr_mins[0])
    
    return found_lines