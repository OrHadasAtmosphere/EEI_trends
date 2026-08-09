#!/bin/bash
set -e
set -u

echo "Using code environment from .yaml"
micromamba env create -f environment.yaml
eval "$(micromamba shell hook --shell bash)"
micromamba activate eei-env
echo "Installing tools"
pip install .

echo "Processing CERES EBAF product to linear trends and global means"
python -m obs_io.ceres

echo "Downloading ERA5 data"
python -m obs_io.download_era5_drivers
python -m obs_io.download_era5_masks

echo "Defining masks for dynamical regimes based on ERA5 climatology"
python -m obs_io.process_era5clim
python -m obs_io.process_masks

echo "Processing CERES EBAF trends over dynamical regimes"
python -m obs_io.ceres_regimes

echo "Processing drivers to trends"
python -m obs_io.process_CRH
python -m obs_io.process_drivers

echo "Computing analytic clear-sky OLR trends for comparison"
python -m obs_io.OLR_analytic

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
