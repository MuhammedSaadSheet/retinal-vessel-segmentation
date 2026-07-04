import argparse
from pathlib import Path

from retinal_vessels.config import load_config
from retinal_vessels.pipeline import run_single_image_pipeline


def parse_args():
    """
    Parse command-line arguments.

    Example:
        python scripts/run_pipeline.py --image data/raw/DRIVE/test/images/01_test.tif
    """
    parser = argparse.ArgumentParser(
        description="Run the retinal vessel segmentation pipeline."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the retinal image.",
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for masks and overlays. If not set, config value is used.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    preprocessing_config = config.get("preprocessing", {})
    segmentation_config = config.get("segmentation", {})
    path_config = config.get("paths", {})

    image_path = Path(args.image)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(path_config.get("output_dir", "results"))

    result_paths = run_single_image_pipeline(
        image_path=image_path,
        output_dir=output_dir,
        image_name=image_path.stem,
        preprocessing_config=preprocessing_config,
        segmentation_config=segmentation_config,
    )

    print("Segmentation completed.")
    print(f"Mask saved to: {result_paths['mask_path']}")
    print(f"Overlay saved to: {result_paths['overlay_path']}")


if __name__ == "__main__":
    main()