from pathlib import Path
import subprocess
import sys


DATASET = "andrewmvd/drive-digital-retinal-images-for-vessel-extraction"
OUTPUT_DIR = Path("data/raw/DRIVE")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(OUTPUT_DIR),
        "--unzip",
    ]

    print("Downloading DRIVE dataset...")
    subprocess.run(command, check=True)

    image_files = list(OUTPUT_DIR.rglob("*.tif")) + list(OUTPUT_DIR.rglob("*.jpg")) + list(OUTPUT_DIR.rglob("*.png"))

    print(f"Done. Found {len(image_files)} image files.")
    for path in image_files[:10]:
        print(path)


if __name__ == "__main__":
    main()