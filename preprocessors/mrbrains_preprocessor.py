import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

class MRBrainSPreprocessor:
    def __init__(self, dataset_path=None, output_path=None):
        self.dataset_path = self._get_dataset_path(dataset_path)
        self.output_path = self._get_output_path(output_path)
        self.modalities = ['T1.nii', 'T1_IR.nii', 'T2_FLAIR.nii']
        self.mask_name = 'LabelsForTraining.nii'

        self._create_directories_if_not_exist(self.output_path)

    def extract_stacked_slices(self):
        subjects = sorted(os.listdir(self.dataset_path))
        for subject in tqdm(subjects, desc="Processing subjects:"):
            subject_dir = os.path.join(self.dataset_path, subject)
            mask_path = os.path.join(subject_dir, self.mask_name)
            mask_volume = self._load_and_align(mask_path)
            modality_volumes = []
            for modality in self.modalities:
                image_path = os.path.join(subject_dir, modality)
                image_volume = self._load_and_align(image_path)
                modality_volumes.append(image_volume)

            multi_modal_volume = np.stack(modality_volumes, axis=-1)    # (H, W, Slices, Modalities)
            self._save_slices(subject_id=subject, image_volume=multi_modal_volume, mask_volume=mask_volume)

    def extract_single_slices(self):
        '''Not recommended. It's better to stack modalities'''
        output_images_path = os.path.join(self.output_path, 'images')
        output_masks_path = os.path.join(self.output_path, 'masks')
        subjects = sorted(os.listdir(self.dataset_path))

        for subject in tqdm(subjects, desc="Processing subjects:"):
            subject_path = os.path.join(self.dataset_path, subject)
            mask_path = os.path.join(subject_path, self.mask_name)
            mask_volume = nib.load(mask_path).get_fdata()
            for modality in self.modalities:
                image_path = os.path.join(subject_path, modality)
                image_volume = nib.load(image_path).get_fdata()
                num_slices = image_volume.shape[2]
                for s in range(num_slices):
                    image_slice = image_volume[:, :, s].astype(np.float32)
                    image_slice_path = f"{output_images_path}/subject{subject}_slice{s}.npy"
                    mask_slice = mask_volume[:, :, s].astype(np.uint8)
                    mask_slice_path = f"{output_masks_path}/subject{subject}_slice{s}.npy"

                    np.save(image_slice_path, image_slice)
                    np.save(mask_slice_path, mask_slice)

    def _save_slices(self, subject_id, image_volume, mask_volume):
        num_slices = image_volume.shape[2]
        for s in range(num_slices):
            image_slice = image_volume[:, :, s, :]  # (H, W, Slice, Modalities)
            mask_slice = mask_volume[:, :, s]

            image_slice_path = os.path.join(self.output_path, 'images', f"subject{subject_id}_slice{s}.npy")
            mask_slice_path = os.path.join(self.output_path, 'masks', f"subject{subject_id}_slice{s}.npy")

            np.save(image_slice_path, image_slice)
            np.save(mask_slice_path, mask_slice)

    def _load_and_align(self, file_path):
        image = nib.load(file_path)
        image = nib.as_closest_canonical(image)
        return image.get_fdata(dtype=np.float32)
    
    def _get_dataset_path(self, dataset_path):
        base_path = os.path.dirname(__file__)
        default_path = os.path.join(base_path, '..', 'data', 'MRBrainS', 'TrainingData')
        return dataset_path if dataset_path is not None else default_path

    def _get_output_path(self, output_path):
        base_path = os.path.dirname(__file__)
        default_path = os.path.join(base_path, '..', 'data', 'MRBrainS', 'preprocessed_2d')
        return output_path if output_path is not None else default_path
    
    def _create_directories_if_not_exist(self, output_path):
        image_out_dir_path = os.path.join(output_path, 'images')
        mask_out_dir_path = os.path.join(output_path, 'masks')

        os.makedirs(image_out_dir_path, exist_ok=True)
        os.makedirs(mask_out_dir_path, exist_ok=True)