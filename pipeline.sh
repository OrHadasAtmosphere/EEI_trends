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
python -m obs_io.download_era5_regimes

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
# main figs
python -m make_figs.net_eei
python -m make_figs.barchart
python -m make_figs.trends_per_regime
python -m make_figs.eei_and_drivers
# supplement figs
python -m make_figs.all_eei_maps
python -m make_figs.masks
python -m make_figs.stcu
python -m make_figs.clr_OLR_analytic

echo "Checking diff. clim. years"
python -m obs_io.diff_yrs
python -m make_figs.diff_yrs
python -m make_figs.regime_changes
