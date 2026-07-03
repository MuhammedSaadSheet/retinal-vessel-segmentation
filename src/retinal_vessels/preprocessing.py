from pathlib import Path

import cv2
import numpy as np
from skimage import img_as_float


def load_image(image_path):
    """
    Load an image from the specified path and convert it to RGB format.

    Parameters:
        image_path (str or Path): Path to the image file.

    Returns:
        numpy.ndarray: Loaded image in RGB format.
    """
    image_path = Path(image_path)

    # OpenCV loads images in BGR format by default.
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    # Check immediately whether the image was loaded correctly.
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert from BGR to RGB because RGB is easier to interpret
    # and is commonly used by visualization libraries.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def extract_green_channel(image_rgb):
    """
    Extract the green channel from an RGB image.

    In retinal fundus images, blood vessels are usually most visible
    in the green channel because the contrast between vessels and
    background is stronger there.

    Parameters:
        image_rgb (numpy.ndarray): Input RGB image.

    Returns:
        numpy.ndarray: Green channel as a grayscale image.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel RGB image.")

    # RGB channels:
    # channel 0 = red
    # channel 1 = green
    # channel 2 = blue
    green_channel = image_rgb[:, :, 1]

    return green_channel


def enhance_contrast(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhance the contrast of a grayscale image using CLAHE.

    CLAHE stands for Contrast Limited Adaptive Histogram Equalization.
    It improves local contrast and is useful for retinal images because
    illumination is often uneven.

    Parameters:
        gray_image (numpy.ndarray): Grayscale input image.
        clip_limit (float): Contrast limiting value for CLAHE.
        tile_grid_size (tuple): Size of the local CLAHE regions.

    Returns:
        numpy.ndarray: Contrast-enhanced image as float image in range [0, 1].
    """
    if gray_image.ndim != 2:
        raise ValueError("CLAHE expects a 2D grayscale image.")

    # OpenCV CLAHE expects uint8 input.
    # If the image is not uint8, normalize it to the range [0, 255].
    if gray_image.dtype != np.uint8:
        gray_image = cv2.normalize(
            gray_image,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tuple(tile_grid_size),
    )

    enhanced_image = clahe.apply(gray_image)

    # Convert to float in range [0, 1] for later scikit-image processing.
    enhanced_float = img_as_float(enhanced_image)

    return enhanced_float


def preprocess_image(image_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Run the complete preprocessing pipeline for one retinal image.

    Steps:
        1. Extract green channel
        2. Apply CLAHE contrast enhancement
        3. Return normalized enhanced image

    Parameters:
        image_rgb (numpy.ndarray): Input RGB retinal image.
        clip_limit (float): CLAHE contrast limit.
        tile_grid_size (tuple): CLAHE tile grid size.

    Returns:
        numpy.ndarray: Preprocessed grayscale image.
    """
    green_channel = extract_green_channel(image_rgb)

    enhanced_image = enhance_contrast(
        green_channel,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )

    return enhanced_image