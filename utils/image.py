"""Transform PIL images into torch tensors"""

from typing import List, Callable
from enum import Enum

import torch
from torchvision.transforms import v2
from PIL import Image


class TransformationLevel(Enum):
    """
    All the levels you can use
    for image transformation pipeline.
    It's useful for testing.
    """

    ALL = 5
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


def get_transformations_pipeline(
    width: int, height: int, level: TransformationLevel
) -> List[Callable]:
    """
    Returns a transformation pipeline
    based on the level.
    """

    transformations = [
        v2.Resize((width, height), interpolation=Image.LANCZOS),
        v2.PILToTensor(),
        v2.CenterCrop(size=(height - 90, width - 20)),
        v2.JPEG((10, 70)),
        v2.ToDtype(torch.float16, scale=True),
    ]

    return transformations[: level.value]


def transform_image(
    img: Image,
    width: int,
    height: int,
    transform_level: TransformationLevel = TransformationLevel.ALL,
) -> torch.Tensor:
    """Transform and normalize a PIL image into a torch tensor ranging values from 0 to 1"""

    image_tensor = v2.Compose(
        get_transformations_pipeline(width, height, transform_level)
    )(img)

    # by default the returning tensor dtype is torch.float32
    # but using TransformationLevel.ALL, it's remapped ot float16
    return image_tensor / 255.0


def look_transformation(img_path: str, width: int, height: int):
    """Test multiple sizes for images"""

    import matplotlib.pyplot as plt

    with Image.open(img_path) as img:
        img = Image.open(img_path)
        transformed_img = transform_image(img, width, height)

        for i in range(3):
            ax = plt.subplot(1, 3, i + 1)
            ax.imshow(transformed_img[i])

        plt.show()
