#!/bin/bash
set -e
set -u

echo "Using code environment from .yaml"
micromamba env create -f environment.yaml
eval "$(micromamba shell hook --shell bash)"
micromamba activate eei-env
echo "Installing tools"
pip install .

echo "Downloading ERA5 data"
python -m obs_io.download_era5_drivers
python -m obs_io.download_era5_regimes

echo "Processing CERES EBAF product to linear trends and global means"
python -m obs_io.ceres

echo "Processing ERA5 climatology to trends on CERES grid"
python -m obs_io.process_era5clim

echo "Processing drivers to trends on CERES grind"
python -m obs_io.process_drivers

echo "Defining masks for dynamical regimes based on climatology"
python -m obs_io.process_regimes

echo "Processing CERES EBAF trends over regimes"
python -m obs_io.ceres_regimes

echo "Making plots"
for fig in make_figs/*.py; do    
    module_name=$(basename "$fig" .py)

    # skip analysis of effect using of different subsets
    # of CERES record and years of ERA5 on results
    if [ "$module_name" = "diff_yrs" ]; then
        echo "Skipping: $fig"
        continue
    fi
    
    echo "Running: python -m make_figs.$module_name"
    python -m make_figs.$module_name
done