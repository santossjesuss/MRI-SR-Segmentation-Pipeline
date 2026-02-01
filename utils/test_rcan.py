import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T
from models.rcan.rcan import RCAN
from gpu import enable_cuda

def test_rcan(img_path):
    device = enable_cuda()

    lr_img = Image.open(img_path)
    transform = T.ToTensor()
    lr_imgTensor = transform(lr_img).unsqueeze(0).to(device)

    rcan = RCAN().to(device)

    sr_tensor = rcan(lr_imgTensor)
    sr_img = sr_tensor.squeeze(0)

    to_pil_image(sr_img).show()