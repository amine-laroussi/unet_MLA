
import importlib
import os
import numpy as np
import random
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.ndimage import distance_transform_edt
from dataset_normalizer import data_augmentation
# --------------------------------------------------
#  1. Fonction Center Crop Image
# --------------------------------------------------

def center_crop_img(feature_map, target_tensor_shape):
    """ Perform edge clipping in a centered manner."""
    h, w = feature_map.shape
    th, tw = target_tensor_shape

    delta_h = h - th
    delta_w = w - tw

    top = delta_h // 2
    left = delta_w // 2

    return feature_map[top:top+th, left:left+tw]


# ========== WEIGHTED CROSS ENTROPY LOSS ======================================
def unet_weight_map(mask, w0=10, sigma=5):
    """
    Compute a pixel-wise weight map following the U-Net (2015) formulation.
    Used in a weighted cross-entropy loss to handle class imbalance
    and emphasize boundaries between touching objects.
    """
    labels = mask.astype(np.int32)

    # ---- Class balancing term w_c(x)
    classes, counts = np.unique(labels, return_counts=True)
    freq = counts / counts.sum()
    alpha = 1.0 / freq                      # inverse class frequency
    alpha_map = dict(zip(classes, alpha))

    w_c = np.zeros_like(labels, dtype=np.float32)
    for c, w in alpha_map.items():
        w_c[labels == c] = w

    # ---- Distance-based boundary weighting (U-Net)
    if labels.max() < 2:
        return np.ones_like(labels, dtype=np.float32)

    distances = []
    for label_id in range(1, labels.max() + 1):
        distances.append(distance_transform_edt(labels != label_id))

    distances = np.stack(distances)
    d1 = np.min(distances, axis=0)           # closest object
    d2 = np.partition(distances, 1, axis=0)[1]  # second closest object

    # ---- Final weight map
    weight = w_c + w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))
    return weight.astype(np.float32)
# ========== DATASET DEFINITION ================================================
class SegmentationDataset(Dataset):
    def __init__(self, img2mask, train=True):
        self.img2mask = img2mask
        self.images = list(img2mask.keys())
        self.train = train
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieving paths
        img_path = self.images[idx]
        mask_path = self.img2mask[img_path]

        # Retrieving images and masks
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Convert to numpy arrays
        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        # Data augmentations
        if self.train:
            importlib.reload(data_augmentation)
            image, mask = data_augmentation.elastic_deformation_3x3(image, mask)
            image, mask = data_augmentation.random_rotate_shift(image, mask)
            image = data_augmentation.intensity_variation(image)

        # Resize mask to target size for UNet
        mask = center_crop_img(mask, (388, 388))

        # Transformations to tensors
        image = self.to_tensor(image / 255.0).float()

        # --- Prepare mask for weight map generation ---
        mask_np_for_wmap = mask.copy()  # copy original mask as numpy array
        mask_np_for_wmap = mask_np_for_wmap.astype(np.int32)  # convert to int32
        # Keep 0/255 values for distance-based weight calculation
        weight_map = unet_weight_map(mask_np_for_wmap)  # numpy float32
        weight_map = torch.from_numpy(weight_map).float()  # convert to tensor float

        # --- Binarize mask for the network ---
        mask = (mask == 255).astype(np.int32)  # binarize
        mask = torch.from_numpy(mask).long()  # convert to tensor long

        return (image, mask, weight_map)