import os
from typing import List

import pytest
from PIL import Image
import torch

from utils.image import (
    transform_image,
    get_transformations_pipeline,
    TransformationLevel,
)


@pytest.fixture()
def image_path() -> str:
    """return test image path"""
    return os.path.join(".", "tests", "test_image.png")


@pytest.fixture()
def image_obj(image_path) -> Image:
    """return a PIL image"""
    return Image.open(image_path)


@pytest.fixture()
def transforms_names() -> List[str]:
    """
    Return a list that contains the names of the
    transformations.
    """

    return ["Resize", "PILToTensor", "CenterCrop", "JPEG", "ToDtype"]


class TestTransformImage:
    """
    Test images transformations.
    """

    def test_get_transformations_for_level_one(self, transforms_names):
        """Should return a single image transformation"""

        transformations = get_transformations_pipeline(
            100, 100, TransformationLevel.ONE
        )

        assert len(transformations) == 1
        assert type(transformations[0]).__name__ == transforms_names[0]

    def test_get_transformations_for_level_two(self, transforms_names):
        """Should return two image transformations"""

        transformations = get_transformations_pipeline(
            100, 100, TransformationLevel.TWO
        )

        t_names = [type(t).__name__ for t in transformations]
        assert t_names == transforms_names[:2]

    def test_get_transformations_for_level_three(self, transforms_names):
        """Should return three image transformations"""

        transformations = get_transformations_pipeline(
            100, 100, TransformationLevel.THREE
        )
        t_names = [type(t).__name__ for t in transformations]
        assert t_names == transforms_names[:3]

    def test_get_transformations_for_level_four(self, transforms_names):
        """Should return four image transformations"""

        transformations = get_transformations_pipeline(
            100, 100, TransformationLevel.FOUR
        )
        t_names = [type(t).__name__ for t in transformations]
        assert t_names == transforms_names[:4]

    def test_get_transformations_for_level_all(self, transforms_names):
        """Should return five image transformations"""
        transformations = get_transformations_pipeline(
            100, 100, TransformationLevel.ALL
        )
        t_names = [type(t).__name__ for t in transformations]

        assert t_names == transforms_names

    def test_transformation_right_resize(self, image_obj):
        """Should resize the test image to 100x100 and three channels"""

        t_image = transform_image(image_obj, 100, 100, TransformationLevel.TWO)
        assert t_image.shape == (3, 100, 100)

    def test_center_crop(self, image_obj):
        """
        Should crop the image in the center.
        """
        t_image = transform_image(image_obj, 100, 100, TransformationLevel.THREE)
        assert t_image.shape == (3, 10, 80)

    def test_jpeg(self, image_obj):
        """
        should return an image with the correct dimensions but with
        worse quality.
        """
        t_image = transform_image(image_obj, 100, 100, TransformationLevel.FOUR)
        assert t_image.shape == (3, 10, 80)

    def test_dtype_transform(self, image_obj):
        """
        should map each value to torch.float16.
        """
        t_image = transform_image(image_obj, 100, 100, TransformationLevel.FOUR)
        assert t_image.dtype == torch.float32

        t_image = transform_image(image_obj, 100, 100, TransformationLevel.ALL)
        assert t_image.dtype == torch.float16
