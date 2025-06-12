"""Transform PIL images into torch tensors"""

import torch
from PIL import Image


def transform_image(img: Image, width: int, height: int) -> torch.Tensor:
    """Transform and normalize a PIL image into a torch tensor ranging values from 0 to 1"""

    from torchvision.transforms import v2

    image_tensor = v2.Compose(
        [
            v2.Resize((width, height), interpolation=Image.LANCZOS),
            v2.CenterCrop(size=(height - 90, width - 20)),
            v2.PILToTensor(),
            v2.JPEG((10, 70)),
            v2.ToDtype(torch.float16),
        ]
    )(img)

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
