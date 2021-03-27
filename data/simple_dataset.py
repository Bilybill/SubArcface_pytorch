import os
import numpy as np
import cv2
from torch.utils.data import Dataset

from .augmentation import FaceAugmentationCV2


class SimpleDataset(Dataset):
    def __init__(self, prefix, list_file, crop_size, final_size, crop_center_y_offset):
        with open(list_file) as f:
            self.img_list = [
                os.path.join(prefix, line.strip()) for line in f.readlines()
            ]
        self.img_num = len(self.img_list)

        self.face_aug = FaceAugmentationCV2(
            crop_size, final_size, crop_center_y_offset, 0, 0, -1
        )

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = self.face_aug(img)
        return img
