import os
import random
import numpy as np

import torch

# ID #0 is for CTC blank
CLASSES                  = [' ', '!', '\"', '#', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
CHAR2ID                  = {'~': 0, ' ': 1, '!': 2, '"': 3, '#': 4, '&': 5, "'": 6, '(': 7, ')': 8, '*': 9, '+': 10, ',': 11, '-': 12, '.': 13, '/': 14, '0': 15, '1': 16, '2': 17, '3': 18, '4': 19, '5': 20, '6': 21, '7': 22, '8': 23, '9': 24, ':': 25, ';': 26, '?': 27, 'A': 28, 'B': 29, 'C': 30, 'D': 31, 'E': 32, 'F': 33, 'G': 34, 'H': 35, 'I': 36, 'J': 37, 'K': 38, 'L': 39, 'M': 40, 'N': 41, 'O': 42, 'P': 43, 'Q': 44, 'R': 45, 'S': 46, 'T': 47, 'U': 48, 'V': 49, 'W': 50, 'X': 51, 'Y': 52, 'Z': 53, 'a': 54, 'b': 55, 'c': 56, 'd': 57, 'e': 58, 'f': 59, 'g': 60, 'h': 61, 'i': 62, 'j': 63, 'k': 64, 'l': 65, 'm': 66, 'n': 67, 'o': 68, 'p': 69, 'q': 70, 'r': 71, 's': 72, 't': 73, 'u': 74, 'v': 75, 'w': 76, 'x': 77, 'y': 78, 'z': 79}
ID2CHAR                  = {0: '~', 1: ' ', 2: '!', 3: '"', 4: '#', 5: '&', 6: "'", 7: '(', 8: ')', 9: '*', 10: '+', 11: ',', 12: '-', 13: '.', 14: '/', 15: '0', 16: '1', 17: '2', 18: '3', 19: '4', 20: '5', 21: '6', 22: '7', 23: '8', 24: '9', 25: ':', 26: ';', 27: '?', 28: 'A', 29: 'B', 30: 'C', 31: 'D', 32: 'E', 33: 'F', 34: 'G', 35: 'H', 36: 'I', 37: 'J', 38: 'K', 39: 'L', 40: 'M', 41: 'N', 42: 'O', 43: 'P', 44: 'Q', 45: 'R', 46: 'S', 47: 'T', 48: 'U', 49: 'V', 50: 'W', 51: 'X', 52: 'Y', 53: 'Z', 54: 'a', 55: 'b', 56: 'c', 57: 'd', 58: 'e', 59: 'f', 60: 'g', 61: 'h', 62: 'i', 63: 'j', 64: 'k', 65: 'l', 66: 'm', 67: 'n', 68: 'o', 69: 'p', 70: 'q', 71: 'r', 72: 's', 73: 't', 74: 'u', 75: 'v', 76: 'w', 77: 'x', 78: 'y', 79: 'z'}
   
DEBUG_MODE               = False
   
TRAINED_WEIGHTS_PATH     = ""
   
# INPUT_PATH               = "./input/"  # PC and EC2
INPUT_PATH               = "../../input" # Kaggle
# INPUT_PATH               = "."        # Colab
GENERATED_FILES_PATH     = "./generated/"
# FORMS_PATH               = os.path.join("D:\Kevin\Machine Learning\IAM Dataset Full\original\\forms") # PC
FORMS_PATH               = os.path.join(INPUT_PATH, "iam-handwritten-forms-dataset/data") # Kaggle
DATASET_PATH             = os.path.join(INPUT_PATH, "iam-dataset-modified/upload")
TRAIN_IMAGES_DIR_1       = os.path.join(DATASET_PATH, "train")
TRAIN_IMAGES_DIR_2       = os.path.join(DATASET_PATH, "val")
TRAIN_1                  = os.path.join(DATASET_PATH, "train_list.txt")
TRAIN_2                  = os.path.join(DATASET_PATH, "val_list.txt")
TRAIN_FOLDS              = os.path.join(GENERATED_FILES_PATH, "train_folds.csv")
FORMS_DF                 = os.path.join(GENERATED_FILES_PATH, "forms.csv")
VALIDATE_WEIGHTS_PATH    = os.path.join(INPUT_PATH, "iam-resnet18-weights/HandwrittenTextRecognition-Pytorch/generated/weights/Image2TextNet/Image2TextNet_fold_0_26.bin")
WEIGHTS_PATH             = "./generated/weights/" # For PC and Kaggle
# WEIGHTS_PATH             = "/content/drive/My Drive/weights" # For Colab
# WEIGHTS_PATH             = "/vol/weights/" # For EC2
LENGTH_BIN_SIZE          = 5
MIN_LEN_ALLOWED          = 1
MAX_LEN_ALLOWED          = 70
THRESHOLD                = 0
TIME_STEPS               = 256
RNN_INPUT_SIZE           = 512
   
USE_GPU                  = True # [True, False]
GPUS                     = 1
TPUS                     = 8 # Basically TPU Nodes
SEED                     = 719
FOLDS                    = 5
MIXED_PRECISION_TRAIN    = True # [True, False]
DROP_LAST                = True # [True, False]
DO_FREEZE_BATCH_NORM     = True # [True, False]
FREEZE_BN_EPOCHS         = 5
   
H                        = 1024 # [32, 384, 512]
W                        = 64 # [128, 384, 512]
C                        = 1

CNN_BACKBONE             = "ResNet18" # [None, ResNet18, EfficientNetB0_NS, EfficientNetB1_NS]
BACKBONE_PRETRAINED      = False
OPTIMIZER                = "Adam"  # [Adam, AdamW, RAdam, AdaBelief, RangerAdaBelief]
SCHEDULER                = "CosineAnnealingWarmRestarts" # [ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, StepLR]
SCHEDULER_WARMUP         = True # [True, False]
WARMUP_EPOCHS            = 3 if SCHEDULER_WARMUP else 0
WARMUP_FACTOR            = 7 if SCHEDULER_WARMUP else 1
TRAIN_CRITERION          = "CTCLoss" # [BiTemperedLogisticLoss, LabelSmoothingCrossEntropy, SoftmaxCrossEntropy, FocalCosineLoss, SmoothCrossEntropyLoss, TaylorCrossEntropyLoss, RandomChoice]
VALID_CRITERION          = "CTCLoss" # [BiTemperedLogisticLoss, SoftmaxCrossEntropy, FocalCosineLoss, SmoothCrossEntropyLoss, TaylorCrossEntropyLoss, RandomChoice]
LEARNING_RATE            = 1e-3
MAX_EPOCHS               = 45
SCHEDULER_BATCH_STEP     = True # [True, False]

# ~ is the blank character
N_CLASSES                = len(CLASSES) + 1 # 1 for blank for CTC

TRAIN_BATCH_SIZE         = 32
VALID_BATCH_SIZE         = 32
VALID_SEGMENT_BATCH_SIZE = 1
ACCUMULATE_ITERATION     = 1
   
NET                      = "Image2TextNet" # ["Image2TextNet"]
   
LEARNING_VERBOSE         = True
VERBOSE_STEP             = 1
   
USE_SUBSET               = False
SUBSET_SIZE              = TRAIN_BATCH_SIZE * 1
CPU_WORKERS              = 0

TRAIN_BATCH_SIZE       //= ACCUMULATE_ITERATION
VALID_BATCH_SIZE       //= ACCUMULATE_ITERATION
if USE_GPU:
    TRAIN_BATCH_SIZE   //= GPUS
    VALID_BATCH_SIZE   //= GPUS


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)