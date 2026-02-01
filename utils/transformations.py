import torchvision.transforms as transforms

def transform_image(image):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    return transform(image)