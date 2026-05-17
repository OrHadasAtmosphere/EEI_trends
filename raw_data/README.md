### How to get the data:

#### CERES EBAF 4.2.1:

We use the CERES EBAF 4.2.1 product [described here](https://ceres.larc.nasa.gov/data/#energy-balanced-and-filled-ebaf), and figures used in publication are generated using a `.nc` file containing CERES **TOA and solar fluxes** from 2000-03-01 through 2026-02-28, downloadable [here](https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA421Selection.jsp) on a 1x1 degree grid.

#### OLR lookup table

In our analysis of the LW clear-sky trends in regions of tropical ascent, we use the framework provided by [McKim et al 2021](https://doi.org/10.1029/2021GL094074). Clear-sky outgoing longwave radiation is estimated using a lookup table indexed by surface temperature and upper tropospheric relative humidity found [here](https://zenodo.org/records/5164050#.YQwWf1NKhZ1) titled `zenodo_olr.nc` but titled `OLR_table.nc` in our scripts.