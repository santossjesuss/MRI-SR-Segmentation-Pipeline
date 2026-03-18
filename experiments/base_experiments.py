from abc import ABC, abstractmethod

class BaseExperiments(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_super_resolution(self):
        pass

    @abstractmethod
    def get_hr_segmentation(self):
        pass

    @abstractmethod
    def get_lr_segmentation(self):
        pass

    @abstractmethod
    def get_frozen_sr_frozen_seg(self):
        pass

    @abstractmethod
    def get_frozen_sr_trainable_seg(self):
        pass

    @abstractmethod
    def get_trainable_sr_frozen_seg(self):
        pass

    @abstractmethod
    def get_joint_sr_seg_e2e(self):
        pass

    @abstractmethod
    def get_joint_sr_seg_combined(self):
        pass

    def _get_train_validation_sizes(self, train_dataset_size, train_perc):
        train_size = int(train_perc * train_dataset_size)
        validation_size = train_dataset_size - train_size

        return train_size, validation_size