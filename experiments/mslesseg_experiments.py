from experiments.base_experiments import BaseExperiments
from configs.mslesseg_config import MSLesSegConfig
from datasets.mslesseg_dataset import MSLesSegDataset

class MSLesSegExperiments(BaseExperiments):
    def __init__(self):
        super().__init__(config=MSLesSegConfig(), dataset=MSLesSegDataset)