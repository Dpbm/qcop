from torchvision.transforms import v2
import torch
from PIL import Image
from constants import NEW_DIM

max_width,max_height = NEW_DIM
transform = v2.Compose([
    v2.Resize((max_width, max_height), interpolation=Image.LANCZOS),
    v2.PILToTensor(),
    v2.ToDtype(torch.float16),
])

def transform_image(img):
    return transform(img)/255.0
