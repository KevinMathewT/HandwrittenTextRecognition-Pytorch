import torch.nn as nn
import torch

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


nets = {
    "DETR": DETRModel
}