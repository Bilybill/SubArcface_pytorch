import cv2
import torch
import numpy as np


class FaceAugmentationCV2(object):
    def __init__(
        self, crop_size, final_size, crop_center_y_offset, scale_aug, trans_aug, flip=-1
    ):
        self.crop_size = crop_size
        self.final_size = final_size
        self.crop_center_y_offset = crop_center_y_offset
        self.scale_aug = scale_aug
        self.trans_aug = trans_aug
        self.flip = flip

    def __call__(self, img):

        scale_diff_h = (np.random.rand() * 2 - 1) * self.scale_aug
        scale_diff_w = (np.random.rand() * 2 - 1) * self.scale_aug
        crop_aug_h = self.crop_size * (1 + scale_diff_h)
        crop_aug_w = self.crop_size * (1 + scale_diff_w)

        trans_diff_h = (np.random.rand() * 2 - 1) * self.trans_aug
        trans_diff_w = (np.random.rand() * 2 - 1) * self.trans_aug

        h, w, _ = img.shape
        ct_x = w / 2 * (1 + trans_diff_w)
        ct_y = (h / 2 + self.crop_center_y_offset) * (1 + trans_diff_h)

        if ct_x < crop_aug_w / 2:
            crop_aug_w = ct_x * 2 - 0.5
        if ct_y < crop_aug_h / 2:
            crop_aug_h = ct_y * 2 - 0.5
        if ct_x + crop_aug_w / 2 >= w:
            crop_aug_w = (w - ct_x) * 2 - 0.5
        if ct_y + crop_aug_h / 2 >= h:
            crop_aug_h = (h - ct_y) * 2 - 0.5

        t = int(np.ceil(ct_y - crop_aug_h / 2))
        if ct_y - crop_aug_h / 2 < 1:
            t = 0
        l = int(np.ceil(ct_x - crop_aug_w / 2))
        if ct_x - crop_aug_w / 2 < 1:
            l = 0
        img = img[t : int(np.ceil(t + crop_aug_h)), l : int(np.ceil(l + crop_aug_w)), :]
        img = cv2.resize(img, (self.final_size, self.final_size))

        if np.random.rand() <= self.flip:
            img = cv2.flip(img, 1)

        img = img * 3.2 / 255.0 - 1.6
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        img = img.float()

        return img
