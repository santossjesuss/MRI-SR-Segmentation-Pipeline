class Experiment:
    def __init__(self, name, pipeline, config, training_dataset, validation_dataset, test_dataset):
        self.name = name
        self.pipeline = pipeline
        self.config = config
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

    def run(self):
        print(f"Starting experiment: {self.name}")

        self.name = f'{self.name}.pth'

        pipeline = self.pipeline(config=self.config)
        return pipeline.run(self.training_dataset, self.validation_dataset)

    def test(self):
        print(f"Testing experiment: {self.name}")
        
        pipeline = self.pipeline(config=self.config)
        return pipeline.test(self.test_dataset)