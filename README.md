# Retinal Vessel Segmentation

A professional Python project for retinal blood vessel segmentation in fundus images using classical image processing methods.

The project focuses on preprocessing retinal images, enhancing vessel-like structures, generating binary vessel masks, visualizing segmentation results, and evaluating segmentation quality with standard metrics.

## Project Overview

Retinal blood vessel segmentation is an important task in medical image analysis. Blood vessel structures can provide useful information for screening and analyzing retinal diseases and vascular abnormalities.

This project implements a classical image processing pipeline based on:

```text
RGB fundus image
→ green channel extraction
→ CLAHE contrast enhancement
→ Frangi vessel enhancement
→ Otsu thresholding
→ morphological post-processing
→ binary vessel mask
→ overlay visualization
```

The current implementation does not require deep learning. A future U-Net configuration is included as an extension point.

## Goals

The main goals of this project are:

- Analyze retinal fundus images
- Automatically segment blood vessels
- Generate binary vessel masks
- Create visual overlays of detected vessels
- Evaluate segmentation quality against manual ground truth masks
- Provide a clean, reproducible, and professional GitHub project structure

## Dataset

This project is designed for the DRIVE Retinal Images Dataset.

The dataset is not included in this repository. It should be downloaded separately and placed locally inside the `data/raw/DRIVE/` directory.

Expected dataset structure:

```text
data/raw/DRIVE/
├── training/
│   ├── images/
│   ├── 1st_manual/
│   └── mask/
└── test/
    ├── images/
    ├── 1st_manual/
    └── mask/
```

The dataset is excluded from Git using `.gitignore` because medical image datasets may be large and can have their own usage conditions.

## Project Structure

```text
retinal-vessel-segmentation/
├── .github/
│   └── workflows/
│       └── tests.yml
├── configs/
│   ├── default.yaml
│   └── unet.yaml
├── data/
│   ├── raw/
│   │   └── DRIVE/
│   ├── processed/
│   └── README.md
├── docs/
│   ├── dataset.md
│   └── method.md
├── notebooks/
│   └── 01_exploration.ipynb
├── results/
│   ├── masks/
│   ├── overlays/
│   └── metrics/
├── scripts/
│   ├── run_pipeline.py
│   └── evaluate_dataset.py
├── src/
│   └── retinal_vessels/
│       ├── __init__.py
│       ├── config.py
│       ├── preprocessing.py
│       ├── segmentation.py
│       ├── evaluation.py
│       ├── visualization.py
│       └── pipeline.py
├── tests/
│   ├── test_evaluation.py
│   └── test_preprocessing.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/retinal-vessel-segmentation.git
cd retinal-vessel-segmentation
```

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the environment is not activated, dependencies can also be installed with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

Run the segmentation pipeline on a single retinal image:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\run_pipeline.py --image data\raw\DRIVE\test\images\01_test.tif
```

The generated outputs are saved to:

```text
results/masks/
results/overlays/
```

Example output files:

```text
results/masks/01_test_mask.png
results/overlays/01_test_overlay.png
```

## Configuration

The default classical image processing configuration is stored in:

```text
configs/default.yaml
```

It contains preprocessing and segmentation parameters such as:

```yaml
preprocessing:
  clahe_clip_limit: 2.0
  clahe_tile_grid_size: [8, 8]

segmentation:
  method: frangi
  min_object_size: 80
  hole_area_threshold: 80

paths:
  data_dir: data/raw/DRIVE
  output_dir: results
```

A future deep learning configuration is prepared in:

```text
configs/unet.yaml
```

This file is not used by the current classical pipeline yet. It is included as preparation for a possible U-Net extension.

## Method

The implemented method follows a classical computer vision pipeline.

### 1. Image Loading

Images are loaded with OpenCV and converted from BGR to RGB.

### 2. Green Channel Extraction

The green channel is extracted because retinal blood vessels usually show stronger contrast in this channel.

### 3. Contrast Enhancement

CLAHE is applied to improve local contrast and reduce the effect of uneven illumination.

### 4. Vessel Enhancement

The Frangi filter is used to enhance elongated, vessel-like structures.

### 5. Thresholding

Otsu thresholding converts the vessel response image into a binary mask.

### 6. Morphological Post-processing

Small objects are removed and small holes are filled to clean the vessel mask.

### 7. Visualization

The final binary mask is overlaid on the original retinal image.

## Evaluation Metrics

The project implements the following segmentation metrics:

- Accuracy
- Sensitivity / Recall
- Specificity
- Precision
- Dice Score
- Intersection over Union

These metrics are calculated by comparing the predicted vessel mask with the manual ground truth mask.

## Testing

Run all tests with:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest
```

Expected result:

```text
6 passed
```

The repository also includes a GitHub Actions workflow that automatically runs the tests on push and pull requests to the `main` branch.

## Current Status

Implemented:

- Professional project structure
- YAML configuration loading
- Retinal image preprocessing
- Classical vessel segmentation with Frangi filter
- Binary mask generation
- Overlay visualization
- Evaluation metrics
- Unit tests
- GitHub Actions workflow
- U-Net configuration placeholder

Planned:

- Full DRIVE dataset evaluation script
- CSV export of evaluation metrics
- Example result images
- Notebook-based exploratory analysis
- Optional U-Net implementation

## Technologies

- Python
- OpenCV
- Scikit-Image
- NumPy
- Scikit-Learn
- Pandas
- PyYAML
- Pytest
- GitHub Actions

## License

This project is licensed under the MIT License.