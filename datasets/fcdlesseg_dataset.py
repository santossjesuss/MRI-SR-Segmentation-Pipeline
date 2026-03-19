from datasets.base_seg_dataset import BaseSegmentationDataset

class FCDLesSegDataset(BaseSegmentationDataset):
    def get_default_folder_name(self):
        return 'MRIcontrol_kde'