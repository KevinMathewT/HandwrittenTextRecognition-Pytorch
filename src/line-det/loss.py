from .detr.models.matcher import HungarianMatcher
from .detr.models.detr import SetCriterion
from . import det_config

matcher = HungarianMatcher()
weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}
losses = ['labels', 'boxes', 'cardinality']


def get_train_criterion(device):
    print(f"Training Criterion:          {det_config.TRAIN_CRITERION}")
    loss_map = {
        "BipartiteMatchingLoss": SetCriterion(det_config.NUM_CLASSES - 1, matcher, weight_dict, eos_coef=det_config.NULL_CLASS_COEF, losses=losses),
    }

    return loss_map[det_config.TRAIN_CRITERION]


def get_valid_criterion(device):
    print(f"Validation Criterion:        {det_config.VALID_CRITERION}")

    loss_map = {
        "BipartiteMatchingLoss": SetCriterion(det_config.NUM_CLASSES - 1, matcher, weight_dict, eos_coef=det_config.NULL_CLASS_COEF, losses=losses),
    }

    return loss_map[det_config.VALID_CRITERION]