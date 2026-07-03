import numpy as np
from skimage import filters, morphology


def enhance_vessels_frangi(image):
    """
    Enhance retinal vessels using the Frangi filter.

    The Frangi filter highlights elongated, tube-like structures.
    Retinal blood vessels are such structures, so this filter is useful
    for vessel segmentation.

    Parameters:
        image (numpy.ndarray): Preprocessed grayscale retinal image.

    Returns:
        numpy.ndarray: Vessel response image.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    vessel_response = filters.frangi(image)

    return vessel_response


def threshold_vessels(vessel_response):
    """
    Convert the vessel response image into a binary vessel mask.

    Otsu's method automatically computes a threshold that separates
    foreground vessels from the background.

    Parameters:
        vessel_response (numpy.ndarray): Enhanced vessel response image.

    Returns:
        numpy.ndarray: Binary mask where True means vessel.
    """
    threshold = filters.threshold_otsu(vessel_response)

    binary_mask = vessel_response > threshold

    return binary_mask


def clean_vessel_mask(
    binary_mask,
    min_object_size=80,
    hole_area_threshold=80,
):
    """
    Clean the binary vessel mask.

    Small isolated objects are removed because they are usually noise.
    Small holes inside vessel regions are filled to make the mask smoother.

    Parameters:
        binary_mask (numpy.ndarray): Binary vessel mask.
        min_object_size (int): Minimum object size to keep.
        hole_area_threshold (int): Maximum hole size to fill.

    Returns:
        numpy.ndarray: Cleaned binary vessel mask.
    """
    if binary_mask.ndim != 2:
        raise ValueError("Input binary mask must be a 2D array.")

    cleaned_mask = morphology.remove_small_objects(
        binary_mask.astype(bool),
        min_size=min_object_size,
    )

    cleaned_mask = morphology.remove_small_holes(
        cleaned_mask,
        area_threshold=hole_area_threshold,
    )

    cleaned_mask = morphology.binary_closing(
        cleaned_mask,
        morphology.disk(1),
    )

    return cleaned_mask


def segment_vessels(
    preprocessed_image,
    method="frangi",
    min_object_size=80,
    hole_area_threshold=80,
):
    """
    Segment retinal blood vessels from a preprocessed image.

    Full classical pipeline:
        1. Enhance vessels with Frangi filter
        2. Apply Otsu thresholding
        3. Clean mask with morphology

    Parameters:
        preprocessed_image (numpy.ndarray): Preprocessed grayscale image.
        method (str): Vessel enhancement method. Currently only "frangi".
        min_object_size (int): Minimum object size to keep.
        hole_area_threshold (int): Maximum hole size to fill.

    Returns:
        numpy.ndarray: Final vessel mask with values 0 and 1.
    """
    if method != "frangi":
        raise ValueError("Currently, only the 'frangi' method is supported.")

    vessel_response = enhance_vessels_frangi(preprocessed_image)

    binary_mask = threshold_vessels(vessel_response)

    cleaned_mask = clean_vessel_mask(
        binary_mask,
        min_object_size=min_object_size,
        hole_area_threshold=hole_area_threshold,
    )

    return cleaned_mask.astype(np.uint8)