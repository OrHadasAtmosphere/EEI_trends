import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
from utils import add_weights

season_def = {
    "MAM":[3,4,5],
    "JJA":[6,7,8],
    "SON":[9,10,11],
    "DJF":[12,1,2],
}

# read ERA5 climatology
ds = xr.open_mfdataset(["raw_data/masks_levels.nc","raw_data/masks_pressures.nc"])
ds = ds.rename({"valid_time":"time","latitude":"lat","longitude":"lon"})
ds["omega500"] = ds.sel(pressure_level=500).w
ds = ds.drop_vars(["pressure_level","number","expver","w"])

# proper years and weighting
ds = ds.sel(time=slice("1990-03-01", "2000-02-01"))
ds = add_weights(ds)
ds = ds.sortby(["lat","lon"])

# read SLP variance climatology
slp = xr.open_mfdataset(["antiquated/output/SLP_var_yearly_OnlyTime.nc"])

# make "year" and "month" into one "time" coord
time = pd.to_datetime(
    [f"{y}-{m:02d}-01" for y in slp.year.values for m in slp.month.values]
)
slp = slp.stack(time=("year", "month"))
slp = slp.drop_vars(["year","month","time"])
slp = slp.assign_coords(time=time)
slp = slp.sel(time=slice("1990-03-01", "2000-02-01"))
slp = add_weights(slp)

# regrid to same 1x1
regridder = xe.Regridder(slp, ds, method="bilinear")
slp = regridder(slp)

ds = xr.merge([ds,slp])

# make seasons
all = []
for seas in season_def.keys():
    sub = ds.sel(time=ds.time.dt.month.isin(season_def[seas]))
    sub_mean = sub.weighted(sub.days_in_month).mean("time")
    sub_mean = sub_mean.expand_dims(season=[seas])
    all.append(sub_mean)
ds = xr.concat(all, dim="season")
ds = ds.drop_vars(["days_in_month"])
ds.to_netcdf("pp/era5_clim.nc")
