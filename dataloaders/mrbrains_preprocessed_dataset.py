import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MRBrainSPreprocessedDataset(Dataset):
    def __init__(self):
        self.dataset_path = '/data/MRBrainS/preprocessed_2d'
        self.images = self._load_data(self.dataset_path, isMask=False)
        self.masks = self._load_data(self.dataset_path, isMask=True)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = np.load(self.images[idx])
        mask = np.load(self.masks[idx])

        image = self._normalize(image)

        image = torch.from_numpy(image).unsqueeze(0)    # (H, W) -> (1, H, W)
        mask = torch.from_numpy(mask).long()            # (H, W)

        return image, mask

    def _load_data(self, dataset_path, isMask):
        if (isMask):
            masks_path = os.path.join(dataset_path, 'masks')
            return sorted(os.listdir(masks_path))
        else:
            images_path = os.path.join(dataset_path, 'images')
            return sorted(os.listdir(images_path))
        
    def _normalize(self):
        pass