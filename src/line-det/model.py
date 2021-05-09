import torch.nn as nn
import torch

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from . import det_config


class DETRModel(nn.Module):
    def __init__(self, num_classes=det_config.NUM_CLASSES, num_queries=det_config.NUM_QUERIES, pretrained=det_config.DET_PRETRAINED):
        super(DETRModel, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=pretrained)
        self.in_features = self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)


class EfficientDetection(nn.Module):
    def __init__(self, num_classes=det_config.NUM_CLASSES, image_size=det_config.H, pretrained=det_config.DET_PRETRAINED):
        super(EfficientDetection, self).__init__()
        self.config = get_efficientdet_config('tf_efficientdet_d5')
        self.net = EfficientDet(self.config, pretrained_backbone=pretrained)
        self.config.num_classes = num_classes
        self.config.image_size = image_size
        self.net.class_net = HeadNet(self.config,
                                     num_outputs=config.num_classes,
                                     norm_kwargs=dict(eps=.001, momentum=.01))

    def forward(self, images):
        return self.net(images)


class FasterRCNN(nn.Module):
    def __init__(self, num_classes=det_config.NUM_CLASSES):
        super(FasterRCNN, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.num_classes = num_classes
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)

    def forward(self, images, targets=None):
        if targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)


nets = {
    "DETR": DETRModel,
    "EfficientDet": EfficientDetection,
    "FasterRCNN": FasterRCNN,
}