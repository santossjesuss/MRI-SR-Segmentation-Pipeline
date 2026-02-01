import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T
# from models.unet.unet import UNet
from segmentation_models_pytorch import Unet
from utils.gpu import enable_cuda

def test_unet(img_path, in_channels=3, out_channels=4):
    print("Testing UNet model...")
    device = enable_cuda()

    img = Image.open(img_path).convert('RGB')
    transform = T.ToTensor()
    imgTensor = transform(img).unsqueeze(0).to(device)
    print(f'Input image tensor shape: {imgTensor.shape}')

    # unet = UNet(in_channels, out_channels).to(device)
    unet = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels, 
        classes=out_channels
    ).to(device)
    # unet.load_state_dict(torch.load('unet_segmentation.pth', map_location=device))
    unet.eval()

    seg_result = unet(imgTensor).squeeze(0)

    to_pil_image(seg_result).show()