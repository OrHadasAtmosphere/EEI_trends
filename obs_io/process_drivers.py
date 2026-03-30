import numpy as np
import xarray as xr
from utils.calc_EIS import calc_EIS

def add_weights(ds):
    weights = ds.time.dt.days_in_month
    weights = weights.where(weights.time.dt.month!=2, 28.65)
    ds["days_in_month"] = weights
    return ds

season_def = {
    "ANN":np.arange(12)+1,
    "MAM":[3,4,5],
    "JJA":[6,7,8],
    "SON":[9,10,11],
    "DJF":[12,1,2],
}

# read era5 drivers
ds = xr.open_mfdataset(["raw_data/drivers_levels.nc", "raw_data/drivers_pressures.nc"]) # "raw_data/drivers_levels_2026.nc", "raw_data/drivers_pressures_2026.nc"])
ds["T700"] = ds.sel(pressure_level=700).t
ds["T850"] = ds.sel(pressure_level=850).t
ds = ds.rename({"valid_time":"time","latitude":"lat","longitude":"lon"})
ds["EIS"] = calc_EIS(ds.t2m, ds.sp, ds.T700, ds.T850)
ds = ds.drop_vars(["t","pressure_level","number","expver","T700","T850","sp"])

# proper years and weighting
ds = ds.sel(time=slice("2000-03-01", "2025-03-01"))
ds = add_weights(ds)
year_adj = ds.time.dt.year - (ds.time.dt.month == 1) - (ds.time.dt.month == 2) # define year Mar-Feb
ds = ds.assign_coords(year=year_adj)

# make seasons
all = []
for seas in season_def.keys():
    sub = ds.sel(time=ds.time.dt.month.isin(season_def[seas]))
    sub_mean = (sub*sub.days_in_month).groupby("year").sum("time") / sub.days_in_month.groupby("year").sum("time")
    sub_mean = sub_mean.expand_dims(season=[seas])
    all.append(sub_mean)
ds = xr.concat(all, dim="season")

# calculate trends
ds_trend = ds.polyfit("year",1).sel(degree=1).drop_vars(["degree","days_in_month_polyfit_coefficients"])*10
ds_trend = ds_trend.rename({
    v: v.replace("_polyfit_coefficients", "")
    for v in ds_trend.data_vars
})
ds_trend.to_netcdf("pp/era5_drivers_trends.nc")
