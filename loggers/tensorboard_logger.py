from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger():
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metric(self, metric_name, metric_value, step=None, phase=""):
        tag = f"{phase}/{metric_name}" if phase else metric_name
        if step:
            self.writer.add_scalar(tag, metric_value, step)
        else:
            self.writer.add_scalar(tag, metric_value)

    def log_metrics(self, metrics_dict, step=None, phase=""):
        for name, value in metrics_dict.items():
            tag = f"{phase}/{name}" if phase else name
            if step:
                self.writer.add_scalar(tag, value, step)
            else:
                self.writer.add_scalar(tag, value)

    def close(self):
        self.writer.close()