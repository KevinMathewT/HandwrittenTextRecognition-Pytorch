NET                      = "FasterRCNN"
SEED                     = 719
FOLDS                    = 5
DROP_LAST                = True # [True, False]
USE_GPU                  = True

H                        = 512
W                        = 512
IMAGE_FORMAT             = "pascal_voc" # ["coco", "pascal_voc"]
NORMALIZE_BB             = False
NUM_CLASSES              = 2
NUM_QUERIES              = 15
DET_PRETRAINED           = True
DET_BATCH_SIZE           = 32
NULL_CLASS_COEF          = 0.5
TRAIN_CRITERION          = "BipartiteMatchingLoss"
VALID_CRITERION          = "BipartiteMatchingLoss"
OPTIMIZER                = "Adam"  # [Adam, AdamW, RAdam, AdaBelief, RangerAdaBelief]
SCHEDULER                = "CosineAnnealingWarmRestarts" # [ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, StepLR]
SCHEDULER_WARMUP         = False # [True, False]
WARMUP_EPOCHS            = 1 if SCHEDULER_WARMUP else 0
WARMUP_FACTOR            = 7 if SCHEDULER_WARMUP else 1
LEARNING_RATE            = 5e-4
MAX_EPOCHS               = 20
SCHEDULER_BATCH_STEP     = True # [True, False]
TRAIN_BATCH_SIZE         = 16
VALID_BATCH_SIZE         = 16
DO_FREEZE_BATCH_NORM     = False # [True, False]
FREEZE_BN_EPOCHS         = 5
WEIGHTED_LOSS            = False


LEARNING_VERBOSE         = True
VERBOSE_STEP             = 1
CPU_WORKERS              = 0