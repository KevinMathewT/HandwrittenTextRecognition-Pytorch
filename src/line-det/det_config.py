NET                      = "DETR"
SEED                     = 719
FOLDS                    = 5
DROP_LAST                = True # [True, False]
USE_GPU                  = True

H                        = 512
W                        = 512
NUM_CLASSES              = 1
NUM_QUERIES              = 15
DET_PRETRAINED           = True
DET_BATCH_SIZE           = 32
NULL_CLASS_COEF          = 0.5
TRAIN_CRITERION          = "BipartiteMatchingLoss"
VALID_CRITERION          = "BipartiteMatchingLoss"
OPTIMIZER                = "Adam"  # [Adam, AdamW, RAdam, AdaBelief, RangerAdaBelief]
SCHEDULER                = "CosineAnnealingWarmRestarts" # [ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts, StepLR]
SCHEDULER_WARMUP         = True # [True, False]
WARMUP_EPOCHS            = 3 if SCHEDULER_WARMUP else 0
WARMUP_FACTOR            = 7 if SCHEDULER_WARMUP else 1
LEARNING_RATE            = 1e-3
MAX_EPOCHS               = 30
SCHEDULER_BATCH_STEP     = True # [True, False]
TRAIN_BATCH_SIZE         = 32
VALID_BATCH_SIZE         = 32
DO_FREEZE_BATCH_NORM     = True # [True, False]
FREEZE_BN_EPOCHS         = 5


LEARNING_VERBOSE         = True
VERBOSE_STEP             = 1
CPU_WORKERS              = 0