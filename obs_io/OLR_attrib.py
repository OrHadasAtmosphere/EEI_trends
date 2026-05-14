import xarray as xr
import xesmf as xe
from . import add_weights, march_to_feb_years, seasonal_means, to_trend

drivers = xr.open_mfdataset(["raw_data/drivers_levels.nc", "raw_data/drivers_levels_2026.nc", "raw_data/column_rh.nc"])
drivers = drivers.rename({'valid_time':"time", "latitude":"lat", "longitude":"lon"}).drop_vars(["number","expver"])

ceres = xr.open_dataset("pp/ceres_trends.nc")
ceres = ceres.sortby(["lat","lon"])
regridder = xe.Regridder(drivers, ceres, method="bilinear")
drivers = regridder(drivers)

drivers = drivers.sel(time=slice("2000-03-01", "2026-03-01"))
Ts = drivers.t2m
crh = drivers.column_rh
OLR_table = xr.open_dataset("raw_data/OLR_table.nc")

ds = OLR_table.interp(rh=crh, ts=Ts).drop_vars(["rh","ts"])
ds = add_weights(ds)
ds = march_to_feb_years(ds)
ds = seasonal_means(ds)
tot_lwclr_trend = to_trend(ds).olr
tot_lwclr_trend.to_netcdf()

Ts_mean = Ts.mean("time")
ds = OLR_table.interp(rh=crh, ts=Ts_mean).drop_vars(["rh","ts"])
ds = add_weights(ds)
ds = march_to_feb_years(ds)
ds = seasonal_means(ds)
crh_lwclr_trend = to_trend(ds).olr
crh_lwclr_trend.to_netcdf()

crh_mean = crh.mean("time")
ds = OLR_table.interp(rh=crh_mean, ts=Ts).drop_vars(["rh","ts"])
ds = add_weights(ds)
ds = march_to_feb_years(ds)
ds = seasonal_means(ds)
tot_lwclr_trend = to_trend(ds).olr
tot_lwclr_trend.to_netcdf()