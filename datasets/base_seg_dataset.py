from abc import ABC, abstractmethod
import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from transforms.base_transforms import BaseTransforms

class BaseSegmentationDataset(Dataset, ABC):
    def __init__(self, is_training, scale_factor, dataset_path=None, view='axial'):
        super().__init__()
        self.view = view
        self.is_training = is_training

        self.dataset_path = self._resolve_dataset_path(dataset_path=dataset_path)
        self.images_path = self._get_data_path(is_image=True)
        self.masks_path = self._get_data_path(is_image=False)
        
        self.images_names = self._get_names(data_path=self.images_path)
        self.masks_names = self._get_names(data_path=self.masks_path)
        
        self.transforms = BaseTransforms(scale_factor=scale_factor)

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.images_names[idx])
        mask_path = os.path.join(self.masks_path, self.masks_names[idx])
        
        hr_image = read_image(image_path, mode=ImageReadMode.GRAY)
        lr_image = self.transforms.downsample_image(hr_image)
        hr_mask = read_image(mask_path, mode=ImageReadMode.GRAY)
        lr_mask = self.transforms.downsample_mask(hr_mask)

        hr_image = self.transforms.normalize_image(hr_image)
        lr_image = self.transforms.normalize_image(lr_image)

        hr_mask = hr_mask.squeeze(0)
        lr_mask = lr_mask.squeeze(0)
        hr_mask = self.transforms.normalize_binary_mask(mask=hr_mask)
        lr_mask = self.transforms.normalize_binary_mask(mask=lr_mask)
        
        return hr_image, hr_mask, lr_image, lr_mask

    @abstractmethod
    def get_default_folder_name(self):
        pass

    def _resolve_dataset_path(self, dataset_path):
        if dataset_path is not None:
            return dataset_path
        
        base_path = os.path.dirname(__file__)
        folder_name = self.get_default_folder_name()
        
        return os.path.join(base_path, '..', 'data', folder_name)
    
    def _get_data_path(self, is_image):
        data_type = 'images' if is_image else 'labels'
        suffix = 'Tr' if self.is_training else 'Ts'
        folder_name = f'{data_type}{suffix}'

        return os.path.join(self.dataset_path, folder_name, self.view)
    
    def _get_names(self, data_path):
        return sorted(os.listdir(data_path))