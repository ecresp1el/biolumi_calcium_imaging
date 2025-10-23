# In Vitro Calcium Imaging Analysis

This package is a trimmed version of the Biolumi analysis workflow that starts from precomputed CSV exports instead of raw imaging stacks. It is designed to accompany the dataset that will be hosted on Globus for the Nature Methods submission.

## Repository layout

- `environment.yml` – minimal Conda environment for running the notebook and helper scripts.
- `src/calcium_analysis.py` – reusable helpers for loading session metadata, assembling trial-locked responses, running responsiveness statistics, and generating figures.
- `notebooks/in_vitro_calcium_imaging_analysis.ipynb` – walkthrough notebook that mirrors the publication figures using only the CSV exports.
- `data/` – drop the sensor-specific session folders here (or update the path in the notebook). Each session directory should already contain the processed CSV files created by the original pipeline under `processed_data/processed_image_analysis_output/`.

## Getting started

1. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate in_vitro_calcium_imaging
   ```
2. Place the CSV-based session directories (from the Globus download) inside `data/` or another directory. The notebook exposes a `project_folder` variable that you can point to wherever the session folders live.
3. Open `notebooks/in_vitro_calcium_imaging_analysis.ipynb` in JupyterLab or VS Code and run the cells from top to bottom.

The notebook reproduces the responsiveness quantification and plotting routines without requiring ROI segmentation or TIFF handling.
