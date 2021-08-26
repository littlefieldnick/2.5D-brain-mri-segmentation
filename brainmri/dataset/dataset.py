import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


import albumentations as A
from albumentations.pytorch import ToTensor

def get_preprocessing_fn(encoder, encoder_weights):
    return smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_dataloader(ds, bs=16, shuffle=True):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

def get_augmentations(is_train, apply_transforms=False):
    if is_train and not apply_transforms:
        print("apply_transforms is False. Augmentations not applied")
        
    return A.Compose([
        A.RandomCrop(width = 128, height = 128, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),

        # Pixels
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),
        A.IAAEmboss(p=0.25),
        A.Blur(p=0.01, blur_limit = 3),

        # Affine
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
        ], p=0.8)
    ])

class BrainMriSegmentation(Dataset):
    """
    2.5D Dataset for Brain MRI segmentation
    """

    def __init__(self, data, stack_size=1, transforms=None, preprocessing=None):
        self.data = data
        self.stack_size = stack_size
        self.transforms = transforms
        self.preprocessing = preprocessing

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
            
        if self.preprocessing is not None:
            mask_block = np.reshape(mask_block, (mask_block.shape[0], mask_block.shape[1], 1))
            preprocess = self.preprocessing(image=img_block, mask=mask_block)
            img_block = preprocess["image"]
            mask_block = preprocess["mask"]

        return img_block, mask_block
