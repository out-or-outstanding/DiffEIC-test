import os
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset
from torchvision.transforms import (Compose, Resize, CenterCrop, RandomCrop, ToTensor, Normalize,
                                    RandomHorizontalFlip, ColorJitter)

class LICDatasetWithBGAndText(Dataset):
    def __init__(self,
                 file_list: str,
                 out_size: int = 256,
                 crop_type: str = 'random',
                 use_hflip: bool = False,
                 use_rot: bool = False,
                 use_colorjitter: bool = False):
        self.out_size = out_size
        self.crop_type = crop_type
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        self.use_colorjitter = use_colorjitter
        
        # Load data list
        with open(file_list, 'r') as f:
            self.data_list = [line.strip().split('\t') for line in f.readlines()]
        
        # Define image transforms (仅用于预处理，不包含ToTensor)
        self.transform = self._get_transform()
        
    def _get_transform(self) -> Compose:
        transforms = []
        
        # Cropping
        if self.crop_type == 'random':
            transforms.append(RandomCrop(self.out_size))
        elif self.crop_type == 'center':
            transforms.append(CenterCrop(self.out_size))
        else:
            transforms.append(Resize(self.out_size))
        
        # Horizontal flip
        if self.use_hflip:
            transforms.append(RandomHorizontalFlip(p=0.5))
        
        # Color jitter
        if self.use_colorjitter:
            transforms.append(ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
        
        return Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Data format: image_path\tbackground_path\ttext_description
        img_path, bg_path, text = self.data_list[idx]
        
        # Load images
        jpg_pil = Image.open(img_path).convert('RGB')
        bg_pil = Image.open(bg_path).convert('RGB')
        
        # Apply transforms (不包含ToTensor)
        jpg_pil = self.transform(jpg_pil)
        bg_pil = self.transform(bg_pil)
        
        # Convert PIL to numpy array (NHWC format)
        jpg_np = np.array(jpg_pil)  # [H, W, 3], range [0, 255]
        bg_np = np.array(bg_pil)    # [H, W, 3], range [0, 255]
        
        # Convert to float32 and normalize to [-1, 1] (与原始LICDataset保持一致)
        jpg_np = (jpg_np / 255.0).astype(np.float32)  # [0, 1]
        bg_np = (bg_np / 255.0).astype(np.float32)    # [0, 1]
        
        # Apply augmentations (与原始LICDataset保持一致)
        def augment(img, hflip=False, rotation=False, return_status=False):
            # 简化的数据增强实现
            if hflip and np.random.random() > 0.5:
                img = np.fliplr(img)
            if rotation and np.random.random() > 0.5:
                img = np.rot90(img, k=np.random.choice([1, 2, 3]))
            return img
        
        jpg_np = augment(jpg_np, hflip=self.use_hflip, rotation=self.use_rot)
        bg_np = augment(bg_np, hflip=self.use_hflip, rotation=self.use_rot)
        
        # Final normalization to [-1, 1] (与原始LICDataset保持一致)
        target = (jpg_np * 2 - 1).astype(np.float32)  # [-1, 1]
        source = jpg_np.astype(np.float32)            # [0, 1]
        bg_prior = bg_np.astype(np.float32)           # [0, 1]
        
        return {
            'jpg': target,      # Target image [-1, 1], NHWC format
            'hint': source,     # Source image [0, 1], NHWC format (供control使用)
            'bg': bg_prior,     # Background image [0, 1], NHWC format
            'text': text,         # Text description
            'txt':""
        }