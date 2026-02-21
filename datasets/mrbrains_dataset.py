from email.mime import image
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

class MRBrainSDataset(Dataset):
    def __init__(self, transform=None):
        self.dataset_path = '/data/MRBrainS/TrainingData'
        self.slices = self._load_slices(self.dataset_path)
        # self.samples = self._load_samples()
        self.transform = transform
    
    def _load_slices(self, dataset_path):
        subjects = sorted(os.listdir(dataset_path))
        slices = []
        scans = ['T1.nii', 'T1_IR.nii', 'T1_1mm.nii', 'T2_FLAIR.nii']
        mask = 'LabelsForTraining.nii'

        for subject in subjects:
            subject_path = os.path.join(self.dataset_dir, subject)
            mask_path = os.path.join(subject_path, mask)
            for scan in scans:
                image_path = os.path.join(subject_path, scan)
                image = nib.load(image_path)
                num_slices = image.shape[2] # (H, W, D) -> D
                for s in range(num_slices):
                    slices.append((image_path, mask_path, s))
        
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        image_path, mask_path, slice_idx = self.slices[idx]

        image_volume = nib.load(image_path)
        mask_volume = nib.load(mask_path)

        image = np.array(image_volume.dataobj[:, :, slice_idx]).astype(np.float32)
        mask = np.array(mask_volume.dataobj[:, :, slice_idx]).astype(np.float32)

        image = self._normalize(image)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).long

        return image, mask

    # def __getitem__(self, idx):
    #     image_path, mask_path, slice_idx = self.slices[idx]

    #     image_volume = nib.load(image_path).get_fdata
    #     mask_volume = nib.load(mask_path).get_fdata()

    #     image = image_volume[:, :, slice_idx].astype(np.float32)
    #     mask = mask_volume[:, :, slice_idx].astype(np.float32)

    #     image = self._normalize(image)

    #     if self.transform:
    #         augmented = self.transform(image=image, mask=mask)
    #         image = augmented['image']
    #         mask = augmented['mask']
        
    #     image = torch.from_numpy(image).unsqueeze(0)
    #     mask = torch.from_numpy(mask).long

    #     return image, mask

    # def _load_samples(self):
    #     subjects = sorted(os.listdir(self.dataset_dir))
    #     samples = []
    #     for subject in subjects:
    #         subject_dir = os.path.join(self.dataset_dir, subject)

    #         t1 = nib.load(os.path.join(subject_dir, "T1.nii")).get_fdata()
    #         t1_ir = nib.load(os.path.join(subject_dir, "T1_IR.nii")).get_fdata()
    #         t1_flair = nib.load(os.path.join(subject_dir, "T1_FLAIR.nii")).get_fdata()
    #         t1_1mm = nib.load(os.path.join(subject_dir, "T1_1mm.nii")).get_fdata()
    #         mask = nib.load(os.path.join(subject_dir, "LabelsForTraining.nii")).get_fdata()

    #         image = np.stack([t1, t1_ir, t1_flair, t1_1mm], axis=0)

    #         image = self._normalize(image)

    #         depth = image.shape[-1]
    #         for z in range(depth):
    #             img_slice = image[..., z]
    #             mask_slice = mask[..., z]
    #             samples.append((img_slice, mask_slice)) 
        
    #     return samples
    
    # def __len__(self):
    #     return len(self.samples)

    # def __getitem__(self, idx):
    #     image, mask = self.samples[idx]

    #     if self.transform:
    #         augmented = self.transform(image=image, mask=mask)
    #         image = augmented['image']
    #         mask = augmented['mask']
        
    #     image = torch.from_numpy(image).float()
    #     mask = torch.from_numpy(mask).long()
        
    #     return image, mask
    
    def _normalize(self, image):
        pass