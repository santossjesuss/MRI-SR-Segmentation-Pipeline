import torch

def save_model_for_inference(model, saving_name):
    torch.save(
        model.state_dict(), 
        saving_name
    )
    print(f'\t New best model saved')

def load_model_for_inference(model, saving_name):
    model.load_state_dict(torch.load(saving_name))

def save_model_checkpoint(self, epoch, validation_name, validation_score):
    torch.save({
        'epoch': epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'validation_score': validation_score
    }, self.saving_name)

    print(f'\tNew best model saved with {validation_name}: {validation_score:.4f}')

def load_model_checkpoint(self):
    checkpoint = torch.load(self.saving_name)
    self.model.load_state_dict(checkpoint['model_state_dict'])