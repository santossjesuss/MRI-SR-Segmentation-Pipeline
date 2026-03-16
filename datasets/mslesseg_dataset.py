import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from transforms.base_transforms import BaseTransforms

class MSLesSegDataset(Dataset):
    def __init__(self, isTraining, scale_factor, dataset_path=None, view='axial'):
        dataset_path = self._get_dataset_path(dataset_path)
        self.view = view
        self.images_path = self._get_images_path(dataset_path, isTraining, view=self.view)
        self.masks_path = self._get_masks_path(dataset_path, isTraining, view=self.view)
        self.images_names = self._get_names(self.images_path, isTraining)
        self.masks_names = self._get_names(self.masks_path, isTraining)
        self.transforms = BaseTransforms(scale_factor)

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

    def _get_dataset_path(self, dataset_path):
        if dataset_path is not None:
            return dataset_path
        
        base_path = os.path.dirname(__file__)
        default_path = os.path.join(base_path, '..', 'data', 'MRIms_kde')
        
        return default_path
    
    def _get_data_path(self, dataset_path, isTraining, view, isImage):
        data_type = 'images' if isImage else 'labels'

        if isTraining:
            folder_name = f'{data_type}Tr'
        else:
            folder_name = f'{data_type}Ts'

        return os.path.join(dataset_path, folder_name, view)
    
    def _get_images_path(self, dataset_path, isTraining, view):
        return self._get_data_path(dataset_path=dataset_path, isTraining=isTraining, view=view, isImage=True)

    def _get_masks_path(self, dataset_path, isTraining, view):
        return self._get_data_path(dataset_path=dataset_path, isTraining=isTraining, view=view, isImage=False)
    
    def _get_names(self, data_path, isTraining):
        if isTraining:
            names = sorted(os.listdir(data_path))
        else:
            names = sorted(os.listdir(data_path))

        return names