import os
import numpy as np
import nibabel as nib
import tqdm

def extract_slices(dataset_path=None, output_path=None):
    if dataset_path == None:
        dataset_path = '/data/MRBrainS/TrainingData'

    if output_path == None:
        output_path = '/data/MRBrainS/preprocessed_2d'

    subjects = sorted(os.listdir(dataset_path))
    scans = ['T1.nii', 'T1_IR.nii', 'T2_FLAIR.nii']
    mask = 'LabelsForTraining.nii'
    output_images_path = os.path.join(output_path, 'images')
    output_masks_path = os.path.join(output_path, 'masks')

    for subject in tqdm(subjects, desc="Processing subjects:"):
        subject_path = os.path.join(dataset_path, subject)
        mask_path = os.path.join(subject_path, mask)
        mask_volume = nib.load(mask_path).get_fdata()
        for scan in scans:
            image_path = os.path.join(subject_path, scan)
            image_volume = nib.load(image_path).get_fdata()
            num_slices = image_volume.shape[2]
            for s in range(num_slices):
                image_slice = image_volume[:, :, s].astype(np.float32)
                image_slice_path = f"{output_images_path}/subject{subject}_slice{s}.npy"
                mask_slice = mask_volume[:, :, s].astype(np.uint8)
                mask_slice_path = f"{output_masks_path}/subject{subject}_slice{s}.npy"

                np.save(image_slice_path, image_slice)
                np.save(mask_slice_path, mask_slice)