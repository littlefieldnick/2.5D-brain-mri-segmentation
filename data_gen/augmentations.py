import albumentations as A
from albumentations.pytorch import ToTensor

def get_augmentations(transforms=None):
    if transforms is None:
        return A.Compose([
            ToTensor()
        ])

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(width=256, height=256, p=1.0),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.05, rotate_limit=0, p=0.25),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomRotate90(),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8),
        A.ToTensor()
    ])