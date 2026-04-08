import numpy as np
import xarray as xr

### 
# make regime masks
###

ds = xr.open_dataset("pp/era5_clim.nc")

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
TROPICAL_LAT = 40
POLAR_LAT = 60
SIC = 0.1
ASCENT_OMEGA = 0
SUBSIDENCE_OMEGA = 0
NH_STORMS_SLP = 20
SH_STORMS_SLP = 30

ds["nh_cryosphere"] = ds.nh_cryosphere.where(((ds.lat >= POLAR_LAT) & (ds.siconc > SIC)) # sea ice >60
                                    | ((ds.lat >=75) & (ds.lsm > 0.5)) # any land >75 
                                    | ((ds.lat >=POLAR_LAT) & (ds.lsm > 0.5) & (ds.lon > 300) & (ds.lon < 350)), 0) # greenland
ds["sh_cryosphere"] = ds.sh_cryosphere.where(((ds.lat <= -POLAR_LAT) & ((ds.siconc > SIC) # sea ice >60
                                    | (ds.lsm > 0.5))) | (ds.lat <= -70), 0) # any land >70
ds["nh_storms"] = ds.nh_storms.where((ds.lat > 0) & (ds.SLP_var > NH_STORMS_SLP) & (ds.nh_cryosphere < 0.5), 0)
ds["sh_storms"] = ds.sh_storms.where((ds.lat < 0) & (ds.SLP_var > SH_STORMS_SLP) & (ds.sh_cryosphere < 0.5), 0)
ds["tropical_ascent"] = ds.tropical_ascent.where((ds.lat >= -TROPICAL_LAT) & (ds.lat <= TROPICAL_LAT) 
                                    & (ds.omega500 < ASCENT_OMEGA) 
                                    & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5), 0)
ds["subsidence_land"] = ds.subsidence_land.where((ds.lat >= -TROPICAL_LAT) & (ds.lat <= TROPICAL_LAT) 
                                    & (ds.omega500 > SUBSIDENCE_OMEGA) 
                                    & (ds.lsm > 0.5) & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5), 0)
ds["subsidence_ocean"] = ds.subsidence_ocean.where((ds.lat >= -TROPICAL_LAT) & (ds.lat <= TROPICAL_LAT) 
                                    & (ds.omega500 > SUBSIDENCE_OMEGA) 
                                    & (ds.lsm < 0.5) & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5), 0)
ds["residual"] = ds.residual.where((ds.nh_cryosphere < 0.5) & (ds.sh_cryosphere < 0.5) 
                                    & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5)
                                    & (ds.subsidence_land < 0.5) & (ds.subsidence_ocean < 0.5)
                                    & (ds.tropical_ascent < 0.5), 0)

masks = ds.drop_vars(["lsm","siconc","omega500","SLP_var"])

# add grid-area
R = 6371000  # radius of Earth / m
dlat = np.deg2rad(float(masks.lat.diff("lat").mean()))
dlon = np.deg2rad(float(masks.lon.diff("lon").mean()))
area = (R**2) * dlat * dlon * np.cos(np.deg2rad(masks.lat))
masks["area"] = area.broadcast_like(masks)

# check all points are in exactly one mask
sum = masks.tropical_ascent + masks.subsidence_land + masks.subsidence_ocean + masks.nh_storms + masks.sh_storms + masks.nh_cryosphere + masks.sh_cryosphere + masks.residual
assert np.all(sum == 1), "Not all values are 1"

masks.to_netcdf("pp/regime_masks.nc")
