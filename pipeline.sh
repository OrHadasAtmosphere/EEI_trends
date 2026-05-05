#!/bin/bash
set -e
set -u

echo "Using code environment from .yaml"
micromamba env create -f environment.yaml
eval "$(micromamba shell hook --shell bash)"
micromamba activate
echo "Installing tools"
pip install .

echo "Downloading ERA5 data"
python3 -m obs_io.era5_drivers
python3 -m obs_io.era5_masks

echo "Processing CERES EBAF product to linear trends and global means"
python3 -m obs_io.ceres SAVE_CERES_RAW=True

echo "Processing ERA5 climatology to trends on CERES grid"
python3 -m obs_io.process_era5clim

echo "Processing drivers to trends on CERES grind"
python3 -m obs_io.process_drivers

echo "Defining masks for dynamical regimes based on climatology"
python3 -m obs_io.process_masks

echo "Processing CERES EBAF trends over regimes"
python3 -m obs_io.ceres SAVE_CERES_RAW=False

echo "Making plots"
for fig in make_figs/*.py; do    
    module_name=$(basename "$fig" .py)
    
    echo "Running: python3 -m make_figs.$module_name"
    python3 -m make_figs.$module_name
done