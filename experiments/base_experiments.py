from abc import ABC, abstractmethod

class BaseExperiments(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_super_resolution(self):
        pass

    @abstractmethod
    def get_segmentation(self):
        pass

    @abstractmethod
    def get_frozen_sr_frozen_seg(self):
        pass

    @abstractmethod
    def get_frozen_seg_frozen_sr(self):
        pass

    @abstractmethod
    def get_frozen_sr_trainable_seg(self):
        pass

    @abstractmethod
    def get_trainable_sr_frozen_seg(self):
        pass

    @abstractmethod
    def get_frozen_seg_trainable_sr(self):
        pass

    @abstractmethod
    def get_trainable_seg_frozen_sr(self):
        pass

    @abstractmethod
    def get_joint_sr_seg(self):
        pass

    @abstractmethod
    def get_joint_seg_sr(self):
        pass