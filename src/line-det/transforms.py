import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(bb):
    return A.Compose(
        [
            A.Crop(*bb),
            A.Resize(height=512, width=512, p=1),
            ToTensorV2(p=1.0)
        ],
        p=1.0,
        bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0, label_fields=['labels'])
    )


def get_valid_transforms(bb):
    return A.Compose(
        [
            A.Crop(*bb),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0)
        ],
        p=1.0,
        bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0, label_fields=['labels'])
    )