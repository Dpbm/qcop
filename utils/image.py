"""Transform PIL images into torch tensors"""

from typing import List, Callable

import torch
from torchvision.transforms import v2
from PIL import Image

def get_transformations_pipeline() -> List[Callable]:
    """Returns a transformation pipeline."""

    transformations = [
        v2.Grayscale(),
        v2.PILToTensor(),
        v2.RandomAutocontrast(p=1),
        v2.RandomAdjustSharpness(sharpness_factor=4, p=1),
        v2.ToDtype(torch.float16, scale=True),
    ]

    return transformations


def transform_image(
    img: Image,
    width: int,
) -> torch.Tensor:
    """
    Transform and normalize a PIL image into a torch tensor ranging values from 0 to 1.
    
    The height won't change in any manner.
    """

    img = img.convert("RGB")
    image_tensor = v2.Compose(get_transformations_pipeline())(img) / 255.0
    d, height, out_width = image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]
    
    # pad the image to reach the desired width
    pad = torch.full((d,height,abs(width-out_width)), image_tensor.max(), dtype=torch.float16, device=image_tensor.device)
    image_tensor = torch.cat([image_tensor, pad], dim=2)

    return image_tensor


def look_transformation(img_path: str, width: int):
    """Test multiple sizes for images"""

    import matplotlib.pyplot as plt
    import sys

    with Image.open(img_path) as img:
        img = Image.open(img_path)
        print(f"Input size: {img.size}")

        transformed_img = transform_image(img, width)[0]
        print(f"Size: {sys.getsizeof(transformed_img)/2**10:.2f} Kb")
        print(f"Dimensions: {transformed_img.shape}")

        plt.imshow(transformed_img, cmap="gray")
        plt.show()
