import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MRBrainSPreprocessedDataset(Dataset):
    def __init__(self, dataset_path=None, transform=None):
        self.dataset_path = self._get_dataset_path(dataset_path)
        self.images = self._load_data(self.dataset_path, isMask=False)
        self.masks = self._load_data(self.dataset_path, isMask=True)
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = np.load(self.images[idx])
        mask = np.load(self.masks[idx])

        image = self._normalize(image)
        image, mask = self._apply_transform(image, mask)
        
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()    # (H, W, 1)

        return image, mask

    def _load_data(self, dataset_path, isMask):
        if (isMask):
            masks_path = os.path.join(dataset_path, 'masks')
            return sorted(os.listdir(masks_path))
        else:
            images_path = os.path.join(dataset_path, 'images')
            return sorted(os.listdir(images_path))
        
    def _normalize(self, image):
        for c in range(image.shape[2]):
            mean = image[:, :, c].mean()
            std = image[:, :, c].std()
            image[:, :, c] = (image[:, :, c] - mean) / std
        return image

    def _apply_transform(self, image, mask):
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']        
        return image, mask

    def _get_dataset_path(self, dataset_path):
        base_path = os.path.dirname(__file__)
        default_path = os.path.join(base_path, '..', 'data', 'MRBrainS', 'preprocessed_2d')
        return dataset_path if dataset_path is not None else default_path