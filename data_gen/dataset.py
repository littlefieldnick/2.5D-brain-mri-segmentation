import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensor

def get_dataloader(ds, bs=16, shuffle=True):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

def get_augmentations(is_train, apply_transforms=False):
    if not apply_transforms or not is_train:
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
        ToTensor()
    ])

class BrainMriSegmentation(Dataset):
    """
    2.5D Dataset for Brain MRI segmentation
    """

    def __init__(self, data, stack_size=1, transforms=None):
        self.data = data
        self.stack_size = stack_size
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient_info = self.data.iloc[idx].tolist()
        img_pth = patient_info[1]
        msk_pth = patient_info[2]

        img_block = np.load(img_pth)
        mask_block = np.load(msk_pth) / 255.0
        img_block = np.transpose(img_block, (1, 2, 0))

        if self.transforms is not None:
            transformed = self.transforms(image=img_block, mask=mask_block)
            img_block = transformed["image"]
            mask_block = transformed["mask"]

        return img_block, mask_block