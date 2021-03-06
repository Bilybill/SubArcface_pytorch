import os
import cv2
import numpy as np
import random

import torch
from torch.utils.data import Dataset

import numpy as np

from .augmentation import FaceAugmentationCV2


class FaceDataset(Dataset):
    def __init__(self, config):
        train_data_info = config["train_data"]
        list_file = train_data_info["list"][0]
        meta_file = train_data_info["meta"][0]
        prefix = train_data_info["prefix"][0]
        self.drop_mode = train_data_info["drop_mode"]

        with open(list_file) as f:
            self.img_list = [
                os.path.join(prefix, line.strip()) for line in f.readlines()
            ]
        self.img_num = len(self.img_list)

        with open(meta_file) as f:
            self.meta_list = [int(line.strip()) for line in f.readlines()[1:]]
        assert self.img_num == len(self.meta_list)
        train_data_info["num_classes"] = max(self.meta_list) + 1

        aug_config = config["augmentation"]
        flip = aug_config.get("flip", False)
        # if self.drop_mode:
        #     aug_config["scale_aug"] = 0
        #     aug_config['trans_aug'] = 0
        if flip:
            aug_config["flip"] = 0.5
        else:
            aug_config["flip"] = -1

        self.face_aug = FaceAugmentationCV2(**aug_config)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if (img is None):
            if self.drop_mode:
                random_id = 44644
            else:
                random_id = random.choice(range(len(self.img_list)))
            print("img %s is not available, random_id = %d" % (image_path, random_id))
            return self.__getitem__(random_id)
        h, w, _ = img.shape
        if not self.drop_mode:
            img = self.face_aug(img)
        #img = self.face_aug(img)
        else:
            img = cv2.resize(img, (224, 224))
            img = img * 3.2 / 255.0 - 1.6
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img)
            img = img.float()
        label = self.meta_list[idx]
        return {"image": img, "label": label}
