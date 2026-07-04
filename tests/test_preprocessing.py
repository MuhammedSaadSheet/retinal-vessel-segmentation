import numpy as np

from retinal_vessels.preprocessing import (
    extract_green_channel,
    enhance_contrast,
    preprocess_image,
)


def test_extract_green_channel():
    """
    Test whether the green channel is extracted correctly from an RGB image.
    """
    image_rgb = np.zeros((4, 4, 3), dtype=np.uint8)

    # Set only the green channel to a known value.
    image_rgb[:, :, 1] = 128

    green_channel = extract_green_channel(image_rgb)

    assert green_channel.shape == (4, 4)
    assert np.all(green_channel == 128)


def test_extract_green_channel_rejects_invalid_input():
    """
    Test whether invalid non-RGB input raises an error.
    """
    invalid_image = np.zeros((4, 4), dtype=np.uint8)

    try:
        extract_green_channel(invalid_image)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for non-RGB input."


def test_enhance_contrast_output_shape_and_range():
    """
    Test whether CLAHE returns an image with the same shape
    and float values in the range [0, 1].
    """
    gray_image = np.random.randint(
        0,
        256,
        size=(20, 20),
        dtype=np.uint8,
    )

    enhanced = enhance_contrast(gray_image)

    assert enhanced.shape == gray_image.shape
    assert enhanced.min() >= 0.0
    assert enhanced.max() <= 1.0


def test_preprocess_image_output_is_grayscale():
    """
    Test whether the full preprocessing pipeline returns a 2D grayscale image.
    """
    image_rgb = np.random.randint(
        0,
        256,
        size=(20, 20, 3),
        dtype=np.uint8,
    )

    preprocessed = preprocess_image(image_rgb)

    assert preprocessed.shape == (20, 20)
    assert preprocessed.ndim == 2
    assert preprocessed.min() >= 0.0
    assert preprocessed.max() <= 1.0