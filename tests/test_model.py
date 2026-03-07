import os

from PIL import Image

from utils.image import transform_image
from utils.constants import DEFAULT_NEW_DIM
from train import Model

IMAGES_TESTS_PATH = os.path.join(".", "tests")

class TestModel:
    """
    Test the Model itself.
    """

    def test_image_on_model(self):
        """Check if the images fit in the network."""
        model = Model().half()

        width,height = DEFAULT_NEW_DIM

        small_img = Image.open(os.path.join(IMAGES_TESTS_PATH, "small.png"))
        small_img_tr = transform_image(small_img, width)
        model.forward(small_img_tr)
        
        large_img = Image.open(os.path.join(IMAGES_TESTS_PATH, "large.png"))
        large_img_tr = transform_image(small_img, width)
        model.forward(large_img_tr)
