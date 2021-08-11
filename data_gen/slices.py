import numpy as np
import pandas as pd
import cv2
import os
import shutil

from sklearn.model_selection import train_test_split

class MriStacker():
    """
    2.5D MRI Stack generator. Generates 2.5D stacks with specified stack size, and splits into training, validation, and test sets.

    """
    def __init__(self, root_dir, out_dir, stack_size=3):
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.stack_size = stack_size
        self.train_df, self.valid_df, self.test_df = None, None, None
        self.cleanup_allowed = True

    def _stack_images_masks_flair(self, pth, patient_id, img_cnt):
        """
        Stack the image and masks for specified patient. Uses only the flair channel/
        :param pth: Path to images
        :param patient_id: id of patient to process
        :param img_cnt: number of images to process and stack
        :return: img_stack, msk_stack: Stacked images and masks
        """

        img_stack, msk_stack = [], []
        img_file_name = "{patient}_{id}.tif"
        msk_file_name = "{patient}_{id}_mask.tif"
        for i in range(1, img_cnt + 1):
            img = cv2.imread(os.path.join(pth, img_file_name.format(patient=patient_id, id=i)))
            mask = cv2.imread(os.path.join(pth, msk_file_name.format(patient=patient_id, id=i)))
            img_stack.append(img)
            msk_stack.append(mask)
        img_stack, msk_stack = np.array(img_stack), np.array(msk_stack)
        return img_stack, msk_stack

    def _make_slices(self, img_stacks, mask_stacks, patient_id, out_pth):
        """
        Generate the 2.5D slices for each patient and save to out_pth
        :param img_stacks: Stack of MRI images
        :param mask_stacks: Stack of masks
        :param patient_id: patient id to process
        :param out_pth: Path to save stacks
        :return: None
        """
        img_file_name = "{patient}_{id}_stack"
        msk_file_name = "{patient}_{id}_stack_mask"
        for s in range(1, img_stacks.shape[0] + 1):
            if s < self.stack_size or img_stacks.shape[0] - s <= self.stack_size:
                continue
            slice_idx = np.arange(-1, self.stack_size-1) + s
            im_block = img_stacks[slice_idx,:, :, 1]
            msk_block = mask_stacks[s, :, :, 1] # Output is the mask for the center channel
            np.save(os.path.join(out_pth, img_file_name.format(patient=patient_id, id=s)), im_block)
            np.save(os.path.join(out_pth, msk_file_name.format(patient=patient_id, id=s)), msk_block)

    def _split_by_patients(self, patients, val_split=0.2, test_split=0.1, random_state=42):
        """
        Generate splits by patients.
        :param patients: list of patients
        :param val_split: percentage for validation split
        :param test_split: percentage for test split
        :param random_state: random seed for splits to be reproducible
        :return: train, val, test: training, validation ,testing patient splits
        """
        train, test = train_test_split(patients, test_size=test_split, random_state=random_state)
        train, val = train_test_split(train, test_size=val_split, random_state=random_state)

        return train, val, test

    def gen_train_val_test_split(self, splits=[0.7, 0.2, 0.1], random_state=42):
        """
        Generate training, validation, and test splits based on splits
        :param splits: Percentage of records for training, validation, and test
        :param random_state: random state to reproduce the data
        :return: None
        """
        patient_dirs = os.listdir(self.out_dir)
        msk_list, img_list, patient_list = [], [], []

        for patient in patient_dirs:
            patient_folder = os.path.join(self.out_dir, patient)
            if not os.path.isdir(patient_folder):
                continue

            patient_root = os.path.join(self.out_dir, patient)
            for file in os.listdir(patient_root):
                if "mask" not in file:
                    patient_list.append(patient)
                    img_list.append(os.path.join(patient_root, file))
                    msk_list.append(os.path.join(patient_root, file[:file.find(".npy")] + "_mask.npy"))
        data = pd.DataFrame(data={"Patient": patient_list, "Image": img_list, "Mask": msk_list})
        train, val, test = self._split_by_patients(patient_list, val_split=splits[1], test_split=splits[2], random_state=random_state)

        self.train_df = data[data["Patient"].isin(train)]
        self.valid_df = data[data["Patient"].isin(val)]
        self.test_df = data[data["Patient"].isin(test)]

    def process_patients(self):
        """
        Process patient MRI files and build 2.5D stacks based on stack_size
        :return: None
        """
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        for patient in os.listdir(self.root_dir):
            if ".csv" in patient or ".md" in patient:
                continue
            patient_pth = os.path.join(self.root_dir, patient)
            out_patient_pth = os.path.join(self.out_dir, patient)
            num_imgs = len(os.listdir(patient_pth)) // 2  # Half the length to exclude mask counts
            img_stack, msk_stack = self._stack_images_masks_flair(patient_pth, patient, num_imgs)
            if not os.path.exists(out_patient_pth):
                os.mkdir(out_patient_pth)
            self._make_slices(img_stack, msk_stack, patient, out_patient_pth)

    def cleanup(self):
        """
        Remove generated 2.5D slices. Useful for when running multiple experiments on Google Colab/Kaggle
        :return: None
        """
        if self.cleanup_allowed:
            shutil.rmtree(self.out_dir)
            self.train_df, self.valid_df, self.test_df = None, None, None

    def export_data(self, pth):
        """
        Export the training, validation, and test sets. Assumes the cleanup function is not being used.
        :param pth: Path to save the training, validation, and test sets
        :return: None
        """
        self.cleanup_allowed = False
        self.train_df.to_csv(os.path.join(pth, "train.csv"))
        self.valid_df.to_csv(os.path.join(pth, "valid.csv"))
        self.test_df.to_csv(os.path.join(pth, "test.csv"))
