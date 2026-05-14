import numpy as np
import xarray as xr
import xesmf as xe

from obs_io import add_weights, march_to_feb_years, seasonal_means, to_trend

ceres = xr.open_mfdataset(["pp/ceres_trends.nc"]).sel(season="ANN")
drivers = xr.open_mfdataset(["raw_data/drivers_levels.nc", "raw_data/drivers_levels_2026.nc", "raw_data/rh.nc"])
drivers = drivers.rename({'valid_time':"time", "latitude":"lat", "longitude":"lon"}).drop_vars(["number","expver"])
ceres = ceres.sortby(["lat","lon"])
regridder = xe.Regridder(drivers, ceres, method="bilinear")
drivers = regridder(drivers)

OLR_table = xr.open_dataset("raw_data/OLR_table.nc")

drivers = drivers.sel(time=slice("2000-03-01", "2026-03-01"))
skin = drivers.skt
rh400 = drivers.rh400
meanT = skin.mean("time").expand_dims(time=rh400.time)
meanRH = rh400.mean("time").expand_dims(time=skin.time)

def olr_timeseries_to_lwclr_trend(ds):
    lwclr_ts = seasonal_means(march_to_feb_years(add_weights(ds))) * -1
    return to_trend(lwclr_ts).olr

ds = OLR_table.interp(rh=rh400, ts=meanT).drop_vars(["rh","ts"])
lwclr_varRH = olr_timeseries_to_lwclr_trend(ds).rename("lwclr_varRH")
lwclr_varRH.attrs = {
    "long_name": (
        "RH-component of reconstructed clear-sky LW trend"
    ),
    "units": "W m-2 decade-1",
}

ds = OLR_table.interp(rh=meanRH, ts=skin)
lwclr_varTs = olr_timeseries_to_lwclr_trend(ds).rename("lwclr_varTs")
lwclr_varTs.attrs = {
    "long_name": (
        "Temperature-component of reconstructed clear-sky LW trend"
    ),
    "units": "W m-2 decade-1",
}

ds = OLR_table.interp(rh=rh400, ts=skin).drop_vars(["rh","ts"])
lwclr_varRH_varTs = olr_timeseries_to_lwclr_trend(ds).rename("lwclr_varRH_varTs")
lwclr_varRH_varTs.attrs = {
    "long_name": (
        "Reconstructed clear-sky LW trend"
    ),
    "units": "W m-2 decade-1",
}

out = xr.Dataset({
    "lwclr_varRH": lwclr_varRH,
    "lwclr_varTs": lwclr_varTs,
    "lwclr_varRH_varTs": lwclr_varRH_varTs,
})

out.attrs = {
    "description": (
        "Reconstructed clear-sky longwave trends "
        "derived from McKim et al. (2021) lookup-table interpolation"
    )
}
out.to_netcdf("pp/reconstructed_lwclr_trends.nc")
