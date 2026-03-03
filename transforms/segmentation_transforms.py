import albumentations as A
from albumentations.pytorch import ToTensorV2

def segmentation_train_transform():
    return A.Compose([
        # A.RandomCrop(height=224, width=224, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15, 
            border_mode=0, 
            p=0.5
        ),
        A.ElasticTransform(
            alpha=1, 
            sigma=20, 
            p=0.3
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
    ])

def segmentation_validation_transform():
    return A.Compose([
    ])