from pathlib import Path

from retinal_vessels.preprocessing import load_image, preprocess_image
from retinal_vessels.segmentation import segment_vessels
from retinal_vessels.visualization import create_overlay, save_mask, save_rgb_image


def run_single_image_pipeline(
    image_path,
    output_dir,
    image_name=None,
    preprocessing_config=None,
    segmentation_config=None,
):
    """
    Run the complete vessel segmentation pipeline for one retinal image.

    Steps:
        1. Load image
        2. Preprocess image
        3. Segment vessels
        4. Create overlay
        5. Save mask and overlay

    Parameters:
        image_path (str or Path): Path to the retinal image.
        output_dir (str or Path): Directory where results will be saved.
        image_name (str): Optional output filename prefix.
        preprocessing_config (dict): Preprocessing parameters from config.
        segmentation_config (dict): Segmentation parameters from config.

    Returns:
        dict: Paths to the generated mask and overlay.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if image_name is None:
        image_name = image_path.stem

    preprocessing_config = preprocessing_config or {}
    segmentation_config = segmentation_config or {}

    # 1. Load original retinal image.
    image_rgb = load_image(image_path)

    # 2. Preprocess image:
    #    green channel extraction + CLAHE contrast enhancement.
    preprocessed_image = preprocess_image(
        image_rgb,
        clip_limit=preprocessing_config.get("clahe_clip_limit", 2.0),
        tile_grid_size=preprocessing_config.get("clahe_tile_grid_size", (8, 8)),
    )

    # 3. Segment vessels using the selected method.
    vessel_mask = segment_vessels(
        preprocessed_image,
        method=segmentation_config.get("method", "frangi"),
        min_object_size=segmentation_config.get("min_object_size", 80),
        hole_area_threshold=segmentation_config.get("hole_area_threshold", 80),
    )

    # 4. Create visual overlay.
    overlay = create_overlay(image_rgb, vessel_mask)

    # 5. Define output paths.
    mask_path = output_dir / "masks" / f"{image_name}_mask.png"
    overlay_path = output_dir / "overlays" / f"{image_name}_overlay.png"

    # 6. Save results.
    save_mask(vessel_mask, mask_path)
    save_rgb_image(overlay, overlay_path)

    return {
        "mask_path": mask_path,
        "overlay_path": overlay_path,
    }