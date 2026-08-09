### How to get the data:

#### CERES EBAF 4.2.1:

We use the CERES EBAF 4.2.1 product [described here](https://ceres.larc.nasa.gov/data/#energy-balanced-and-filled-ebaf), and figures used in publication are generated using a `.nc` file containing CERES **TOA and solar fluxes** from 2000-03-01 through 2026-02-28, downloadable [here](https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA421Selection.jsp) on a 1x1 degree grid.

#### ERA5 regime masks:

Run `python -m obs_io.download_era5_masks` to download land and sea-ice data.
The regime-mask processing also expects
`ERA5_omega500_monthly_1991_2026-02.nc` and
`SLP_var_2_10day_1940_2025.nc` in this directory. Raw NetCDF inputs are
excluded from Git.
