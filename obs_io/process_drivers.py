import xarray as xr
import os
import xesmf as xe
from . import add_weights, march_to_feb_years, seasonal_means, to_trend
from utils.calc_EIS import calc_EIS
import warnings

warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered",
    category=RuntimeWarning,
)



inputs = "raw_data/"

ceres = xr.open_dataset("pp/ceres_trends.nc")
ceres = ceres.sortby(["lat","lon"])

# read era5 drivers
ds = xr.open_mfdataset([inputs+"drivers_levels.nc", inputs+"drivers_pressures.nc", inputs+"drivers_levels_2026.nc", inputs+"drivers_pressures_2026.nc", inputs+"column_rh.nc"])
ds = ds.rename({"valid_time":"time","latitude":"lat","longitude":"lon"})
regridder = xe.Regridder(ds, ceres, method="bilinear")
ds = regridder(ds)

ds["T700"] = ds.sel(pressure_level=700).t
ds["T850"] = ds.sel(pressure_level=850).t
ds["EIS"] = calc_EIS(ds.t2m, ds.sp, ds.T700, ds.T850)
ds = ds.drop_vars(["t","pressure_level","number","expver","T700","T850","sp"])

# proper years and weighting
ds = ds.sel(time=slice("2000-03-01", "2026-03-01"))
ds = add_weights(ds)
ds = march_to_feb_years(ds)
ds = seasonal_means(ds)

outfile = "pp/era5_drivers_trends.nc"
if os.path.exists(outfile):
    os.remove(outfile)
to_trend(ds).to_netcdf(outfile)
