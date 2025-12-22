from __future__ import print_function
import os
import cv2
import numpy as np
import random
import pyexr
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import Compose

from PIL import Image, ImageOps, ImageFilter
import torch.nn.functional as F
from einops import rearrange

def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list

class M3D(data.Dataset):
    """The M3D Dataset"""

    def __init__(self, root_dir, list_file, height=504, width=1008, color_augmentation=True,
                 LR_filp_augmentation=True, yaw_rotation_augmentation=True, repeat=1, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir

        self.w = width
        self.h = height

        self.max_depth_meters = 100.0
        self.min_depth_meters = 0.01

        self.color_augmentation = color_augmentation
        self.LR_filp_augmentation = LR_filp_augmentation
        self.yaw_rotation_augmentation = yaw_rotation_augmentation

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.is_training = is_training

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.rgb_depth_list = read_list(list_file)

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):

        # Read and process the image file
        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb = cv2.imread(rgb_name)
        # cv2.imwrite('label_rgb.jpg', rgb)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)
        
        # Read and process the depth file
        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        # gt_depth = cv2.imread(depth_name, -1)
        # gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        # gt_depth = gt_depth.astype(float)/4000
        # gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1

        gt_depth = pyexr.open(depth_name).get()
        gt_depth = gt_depth[:, :, 0]
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1

        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb.copy()

        aug_rgb = self.to_tensor(aug_rgb.copy())

        gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0)).to(torch.float32)

        val_mask = ((gt_depth > 0) & (gt_depth <= self.max_depth_meters)& ~torch.isnan(gt_depth))

        # _min, _max = torch.quantile(gt_depth[val_mask], torch.tensor([0.02, 1 - 0.02]),)
        # gt_depth = gt_depth / 2560.0
        gt_depth_norm = gt_depth / 100.0
        gt_depth_norm = torch.clip(gt_depth_norm, 0.001, 1.0)

        # print(gt_depth_norm.shape)
        # Conduct output
        inputs = {}

        inputs["rgb"] = self.normalize(aug_rgb)
        inputs["gt_depth"] = gt_depth_norm 
        inputs["val_mask"] = val_mask # Valid region; not all-true — unusable areas are excluded; other training datasets use all-true masks (except the projection dataset)
        inputs["mask_100"] = (gt_depth > 0) & (gt_depth <= 100)
        # For this dataset, mask_100 is set to all-true because it cannot be computed. Depth gt beyond 100 meters may include glass/mirror objects; in any case, this dataset is not used for training
        
        # In this dataset, the predicted mask100 should be covered by val_mask, so mask100 theoretically has no effect
        # val_mask controls the region where metrics are computed
        return inputs