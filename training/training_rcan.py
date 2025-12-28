import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..data.super_resolution_dataset import SuperResolutionDataset
from models.rcan.rcan import RCAN

def train_rcan():
    epochs = 100
    learning_rate = 1e-4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    
    base_data_path = '../data/super_resolution_dataset/...'
    dataset = SuperResolutionDataset(base_data_path + '/lr', base_data_path + 'hr')
    dataloader = DataLoader(dataset, shuffle=True)
    
    sr_model = RCAN().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(sr_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_losses = []

    for epoch in range(epochs):
        sr_model.train()
        epoch_loss = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')

        for (lr_image, hr_image) in enumerate(progress_bar):
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)

            optimizer.zero_grad()
            sr_image = sr_model(lr_image)

            loss = criterion(sr_image, hr_image)

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    return sr_model

def evaluate_rcan():
    pass