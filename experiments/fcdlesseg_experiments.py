from experiments.base_experiments import BaseExperiments
from configs.fcdlesseg_config import FCDLesSegConfig
from datasets.fcdlesseg_dataset import FCDLesSegDataset

class FCDLesSegExperiments(BaseExperiments):
    def __init__(self):
        super().__init__(config=FCDLesSegConfig(), dataset=FCDLesSegDataset)