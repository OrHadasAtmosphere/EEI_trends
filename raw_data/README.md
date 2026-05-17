### How to get the data:

#### CERES EBAF 4.2.1:

We use the CERES EBAF 4.2.1 product [described here](https://ceres.larc.nasa.gov/data/#energy-balanced-and-filled-ebaf), and figures used in publication are generated using a `.nc` file containing CERES **TOA and solar fluxes** from 2000-03-01 through 2026-02-28, downloadable [here](https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA421Selection.jsp) on a 1x1 degree grid.

#### OLR lookup table

In our analysis of the tropical LW clear-sky trends, we use the clear-sky OLR look-up table from [McKim et al 2021](https://doi.org/10.1029/2021GL094074). This look-up table was constructed from line-by-line radiative transfer calculations using the PyRADS software for varying surface temperature and relative humidity, assuming a moist-adiabatic profile. The look-up table can be downloaded from [here](https://zenodo.org/records/5164050#.YQwWf1NKhZ1) titled `zenodo_olr.nc` but titled `OLR_table.nc` in our scripts.