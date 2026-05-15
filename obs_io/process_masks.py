import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

### 
# make regime masks
###

infile = ["era5_clim.nc", "diff_yrs/era5_clim_1995.nc", "diff_yrs/era5_clim_2000.nc"]
outfile = ["regime_masks.nc", "diff_yrs/regime_masks_1995.nc", "diff_yrs/regime_masks_2000.nc"]

for fin, fout in zip(infile, outfile):

    ds = xr.open_dataset("pp/"+fin)
    
    # define masks
    ds["nh_cryosphere"] = (ds.lsm>-1)*1
    ds["sh_cryosphere"] = (ds.lsm>-1)*1
    ds["nh_storms"] = (ds.lsm>-1)*1
    ds["sh_storms"] = (ds.lsm>-1)*1
    ds["subsidence_land"] = (ds.lsm>-1)*1
    ds["subsidence_ocean"] = (ds.lsm>-1)*1
    ds["tropical_ascent"] = (ds.lsm>-1)*1
    ds["residual"] = (ds.lsm>-1)*1
    
    # some parameters
    SIGMA_LAT = 1.5
    SIGMA_LON = 1.5
    TROPICAL_LAT = 40
    POLAR_LAT = 60
    SIC_THRESH = 0.1
    LAND_THRESH = 0.1
    OMEGA_THRESH = 0.0
    NH_STORM_FACTOR = 0.2 # TODO
    SH_STORM_FACTOR = 0.3
    
    # smooth dataarray
    def smooth(da):
        return gaussian_filter(
            da,
            sigma=(SIGMA_LAT, SIGMA_LON),
            mode=("nearest", "wrap")  # lat, lon
        )
    
    ds["omega500"] = xr.apply_ufunc(
        smooth,
        ds.omega500,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ds.omega500.dtype],
    )
    ds["SLP_var"] = xr.apply_ufunc(
        smooth,
        ds.SLP_var,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ds.SLP_var.dtype],
    )
    
    
    # define masks
    ds["nh_cryosphere"] = ds.nh_cryosphere.where(((ds.lat >= POLAR_LAT) & (ds.siconc > SIC_THRESH)) # sea ice >60
                                        | ((ds.lat >=75) & (ds.lsm > LAND_THRESH)) # any land >75 
                                        | ((ds.lat >=POLAR_LAT) & (ds.lsm > LAND_THRESH) & (ds.lon > 300) & (ds.lon < 350)), 0) # greenland
    ds["sh_cryosphere"] = ds.sh_cryosphere.where(((ds.lat <= -POLAR_LAT) & (ds.siconc > SIC_THRESH)) # sea ice >60
                                        | ((ds.lat <= -65) & (ds.lsm > LAND_THRESH)), 0) # any land >65
    
    # SLP_var_max for NH and SH
    ds["SLP_var_max_nh"] = ds.SLP_var.where(ds.lat > 0).max(dim=("lat", "lon"))
    ds["SLP_var_max_sh"] = ds.SLP_var.where(ds.lat < 0).max(dim=("lat", "lon"))
    ds["nh_storms"] = ds.nh_storms.where((ds.lat > 0) & (ds.SLP_var > NH_STORM_FACTOR*ds.SLP_var_max_nh) & (ds.nh_cryosphere < 0.5), 0)
    ds["sh_storms"] = ds.sh_storms.where((ds.lat < 0) & (ds.SLP_var > SH_STORM_FACTOR*ds.SLP_var_max_sh) & (ds.sh_cryosphere < 0.5), 0)
    
    ds["tropical_ascent"] = ds.tropical_ascent.where((ds.lat >= -TROPICAL_LAT) & (ds.lat <= TROPICAL_LAT) 
                                        & (ds.omega500 <= -OMEGA_THRESH) 
                                        & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5), 0)
    ds["subsidence_land"] = ds.subsidence_land.where((ds.lat >= -TROPICAL_LAT) & (ds.lat <= TROPICAL_LAT) 
                                        & (ds.omega500 > OMEGA_THRESH) 
                                        & (ds.lsm >= LAND_THRESH) & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5), 0)
    ds["subsidence_ocean"] = ds.subsidence_ocean.where((ds.lat >= -TROPICAL_LAT) & (ds.lat <= TROPICAL_LAT) 
                                        & (ds.omega500 > OMEGA_THRESH) 
                                        & (ds.lsm < LAND_THRESH) & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5), 0)
    
    ds["residual"] = ds.residual.where((ds.nh_cryosphere < 0.5) & (ds.sh_cryosphere < 0.5) 
                                        & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5)
                                        & (ds.subsidence_land < 0.5) & (ds.subsidence_ocean < 0.5)
                                        & (ds.tropical_ascent < 0.5), 0)
    
    masks = ds.drop_vars(["lsm","siconc","omega500","SLP_var","SLP_var_max_nh","SLP_var_max_sh"])
    
    # add grid-area
    R = 6371000  # radius of Earth / m
    dlat = np.deg2rad(float(masks.lat.diff("lat").mean()))
    dlon = np.deg2rad(float(masks.lon.diff("lon").mean()))
    area = (R**2) * dlat * dlon * np.cos(np.deg2rad(masks.lat))
    masks["area"] = area.broadcast_like(masks)
    
    # check all points are in exactly one mask
    sum = masks.tropical_ascent + masks.subsidence_land + masks.subsidence_ocean + masks.nh_storms + masks.sh_storms + masks.nh_cryosphere + masks.sh_cryosphere + masks.residual
    assert np.all(sum == 1), "Not all values are 1"
    
    masks.to_netcdf("pp/"+fout)
