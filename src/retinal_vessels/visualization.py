from pathlib import Path

import cv2
import numpy as np


def create_overlay(image_rgb, mask, color=(255, 0, 0), alpha=0.35):
    """
    Create a colored overlay of the vessel mask on the original RGB image.

    Parameters:
        image_rgb (numpy.ndarray): Original retinal image in RGB format.
        mask (numpy.ndarray): Binary vessel mask with values 0 and 1.
        color (tuple): RGB color used to mark vessel pixels.
        alpha (float): Transparency of the overlay.

    Returns:
        numpy.ndarray: RGB image with the vessel mask overlaid.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel RGB image.")

    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D image.")

    if image_rgb.shape[:2] != mask.shape:
        raise ValueError(
            "Input image and mask must have the same spatial dimensions."
        )

    overlay = image_rgb.copy()

    # Color all pixels that were detected as vessels.
    overlay[mask == 1] = color

    # Blend the original image and the colored overlay.
    blended = cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)

    return blended


def save_mask(mask, output_path):
    """
    Save a binary vessel mask as an image file.

    The internal mask contains values 0 and 1.
    For saving as a visible image, these values are converted to 0 and 255.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_image = (mask * 255).astype(np.uint8)

    cv2.imwrite(str(output_path), mask_image)


def save_rgb_image(image_rgb, output_path):
    """
    Save an RGB image using OpenCV.

    OpenCV expects BGR format when saving images.
    Therefore, the image is converted from RGB to BGR before saving.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_path), image_bgr)