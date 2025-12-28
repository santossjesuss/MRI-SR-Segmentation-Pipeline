import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..data.segmentation_dataset import SegmentationDataset
from models.unet.unet import UNet

def train_unet():
    epochs = 100
    learning_rate = 1e-4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    
    base_data_path = '../data/segmentation_dataset/...'
    dataset = SegmentationDataset(base_data_path + '', base_data_path + '/masks')
    dataloader = DataLoader(dataset, shuffle=True)

    seg_model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(seg_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_losses = []

    for epoch in range(epochs):
        seg_model.train()
        epoch_loss = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')

        for (image, mask) in enumerate(progress_bar):
            image = image.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            predicted_mask = seg_model(image)

            loss = criterion(predicted_mask, mask)

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

    return seg_model

def evaluate_unet(unet, num_samples=5):
    unet.eval()
    device = next(unet.parameters()).device

    dataset = SegmentationDataset()

    with torch.no_grad():
        for i in range(num_samples):
            image, true_mask = dataset[i]
            image = image.unsqueeze(0).to(device)

            predicted_mask = unet(image)
            predicted_mask = torch.sigmoid(predicted_mask) > 0.5

            image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
            true_mask_np = true_mask.squeeze().cpu().numpy()
            pred_mask_np = pred_mask_np.squeeze().cpu().numpy()

            fig, axes = plt.subplot(1, 3, figsize=(12, 4))
            axes[0].imshow(image_np)
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')

            axes[1].imshow(true_mask_np, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(pred_mask_np, cmap='gray')
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')