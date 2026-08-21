### How to get the data:

#### CERES EBAF 4.2.1:

We use the CERES EBAF 4.2.1 product [described here](https://ceres.larc.nasa.gov/data/#energy-balanced-and-filled-ebaf), and figures used in publication are generated using a `.nc` file containing CERES **TOA and solar fluxes** from 2000-03-01 through 2026-02-28, downloadable [here](https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA421Selection.jsp) on a 1x1 degree grid.


#### MODIS AOD:

We use the monthly L3 MODIS Aqua (MYD08_M3) and Terra (MOD08_M3) C6 data.
We use the AOD from the DarkTarget-DeepBlue algorithm: `AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean`.
Files were downloaded from the [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/search/) and individual hdf files were merged into `modis_aod.nc` here.