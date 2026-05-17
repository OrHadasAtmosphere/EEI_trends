import xarray as xr
import xesmf as xe
import pandas as pd
from . import add_weights

season_def = {
    "MAM":[3,4,5],
    "JJA":[6,7,8],
    "SON":[9,10,11],
    "DJF":[12,1,2],
}

ceres = xr.open_dataset("pp/ceres_trends.nc")
ceres = ceres.sortby(["lat","lon"])

# read ERA5 climatology
ds = xr.open_mfdataset(["raw_data/masks_levels.nc","raw_data/masks_pressures.nc"])
ds = ds.rename({"valid_time":"time","latitude":"lat","longitude":"lon"})
ds["omega500"] = ds.sel(pressure_level=500).w
ds = ds.drop_vars(["pressure_level","number","expver","w"])

# read SLP variance climatology
slp = xr.open_mfdataset(["raw_data/SLP_var.nc","raw_data/SLP_var_ext.nc"])

# make "year" and "month" into one "time" coord
time = pd.to_datetime(
    [f"{y}-{m:02d}-01" for y in slp.year.values for m in slp.month.values]
)
slp = slp.stack(time=("year", "month"))
slp = slp.drop_vars(["year","month","time"])
slp = slp.assign_coords(time=time)

# wrap lon, regrid to same 1x1
regridder = xe.Regridder(slp, ceres, method="bilinear")
slp = regridder(slp)
wrapped = xr.concat([ds, ds.isel(lon=0)],dim="lon")
ds = wrapped.assign_coords(lon=list(ds.lon.values) + [ds.lon.values[0] + 360])
regridder = xe.Regridder(ds, ceres, method="bilinear")
ds = regridder(ds)
ds = xr.merge([ds,slp])

# add days-in-month weights
ds_full = add_weights(ds)

# make seasonal clim
ds = ds_full.sel(time=slice("1990-03-01", "2000-03-01"))
all = []
for seas in season_def.keys():
    sub = ds.sel(time=ds.time.dt.month.isin(season_def[seas]))
    sub_mean = sub.weighted(sub.days_in_month).mean("time")
    sub_mean = sub_mean.expand_dims(season=[seas])
    all.append(sub_mean)
ds = xr.concat(all, dim="season")
ds = ds.drop_vars(["days_in_month"])
ds.to_netcdf("pp/era5_clim.nc")

####
# repeat for two other 10-year periods
####
ds = ds_full.sel(time=slice("1995-03-01", "2005-03-01"))
ds = add_weights(ds)
all = []
for seas in season_def.keys():
    sub = ds.sel(time=ds.time.dt.month.isin(season_def[seas]))
    sub_mean = sub.weighted(sub.days_in_month).mean("time")
    sub_mean = sub_mean.expand_dims(season=[seas])
    all.append(sub_mean)
ds = xr.concat(all, dim="season")
ds = ds.drop_vars(["days_in_month"])
ds.to_netcdf("pp/era5_clim_1995.nc")

ds = ds_full.sel(time=slice("2000-03-01", "2010-03-01"))
ds = add_weights(ds)
all = []
for seas in season_def.keys():
    sub = ds.sel(time=ds.time.dt.month.isin(season_def[seas]))
    sub_mean = sub.weighted(sub.days_in_month).mean("time")
    sub_mean = sub_mean.expand_dims(season=[seas])
    all.append(sub_mean)
ds = xr.concat(all, dim="season")
ds = ds.drop_vars(["days_in_month"])
ds.to_netcdf("pp/era5_clim_2000.nc")
