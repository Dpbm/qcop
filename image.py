"""Transform PIL images into torch tensors"""

from torchvision.transforms import v2
import torch
from PIL import Image

def transform_image(img:Image, width:int, height:int) -> torch.Tensor:
    """Transform and normalize a PIL image into a torch tensor ranging values from 0 to 1"""

    image_tensor = v2.Compose([
                        v2.Resize((width, height), interpolation=Image.LANCZOS),
                        v2.PILToTensor(),
                        v2.ToDtype(torch.float16),
                    ])(image)

    return image_tensor/255.0