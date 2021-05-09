import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from . import det_config


def get_train_transforms(bb, format=det_config.IMAGE_FORMAT):
    return A.Compose(
        [
            A.Crop(*bb),
            A.Resize(height=512, width=512, p=1),
            ToTensorV2(p=1.0)
        ],
        p=1.0,
        bbox_params=A.BboxParams(format=format, label_fields=['labels'])
    )


def get_valid_transforms(bb, format=det_config.IMAGE_FORMAT):
    return A.Compose(
        [
            A.Crop(*bb),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0)
        ],
        p=1.0,
        bbox_params=A.BboxParams(format=format, label_fields=['labels'])
    )