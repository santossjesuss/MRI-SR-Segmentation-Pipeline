import os
from PIL import Image
import torchvision.transforms as transforms

class SuperResolutionDataset():
    def __init__(self, low_res_img_dir, high_res_img_dir, transform=None):
        self.low_res_img_dir = low_res_img_dir
        self.high_res_img_dir = high_res_img_dir
        self.transform = transform
        self.images = os.listdir(low_res_img_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        lr_img_path = os.path.join(self.low_res_img_dir, self.images[idx])
        hr_img_path = os.path.join(self.high_res_img_dir, self.images[idx])

        lr_image = Image.open(lr_img_path).convert('RGB')
        hr_image = Image.open(hr_img_path).convert('RGB')

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            lr_image = transform(lr_image)
            hr_image = transform(hr_image)
        
        return lr_image, hr_image