import torch
from torch.utils.data import Dataset

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
        msk_pth = spatient_info[2]

        img_block = np.load(img_pth)
        mask_block = np.load(msk_pth) / 255.0
        img_block = np.transpose(img_block, (1, 2, 0))

        if self.transforms is not None:
            transformed = self.transforms(image=img_block, mask=mask_block)
            img_block = transformed["image"]
            mask_block = transformed["mask"]

        return img_block, mask_block