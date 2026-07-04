# Data Directory

This directory contains the local data files used by the retinal vessel segmentation project.

The DRIVE Retinal Images Dataset is not included in this repository. It must be downloaded separately and placed manually in the expected directory structure.

## Why the dataset is not included

The dataset is excluded from GitHub because:

- medical image datasets may have their own usage conditions
- datasets can be large
- repositories should remain lightweight
- generated or external data should not be version-controlled directly

The `.gitignore` file prevents raw and processed data from being uploaded accidentally.

## Expected Structure

Place the DRIVE dataset under:

```text
data/raw/DRIVE/