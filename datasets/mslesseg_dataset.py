from datasets.base_seg_dataset import BaseSegmentationDataset

class MSLesSegDataset(BaseSegmentationDataset):
    def get_default_folder_name(self):
        return 'MRIms_kde'