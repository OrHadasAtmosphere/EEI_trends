import xarray as xr

from . import to_trend, add_weights, march_to_feb_years, seasonal_means
from utils import global_mean, trend_and_ci

def read_ceres_raw(pathname="raw_data/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202602.nc"):
    ds = xr.open_mfdataset([pathname])
    ds["net"] = ds.toa_net_all_mon
    ds["net_clr"] = ds.toa_net_clr_c_mon
    ds["lw"] = -ds.toa_lw_all_mon
    ds["lw_clr"] = -ds.toa_lw_clr_c_mon
    ds["sw"] = ds.solar_mon - ds.toa_sw_all_mon
    ds["sw_clr"] = ds.solar_mon - ds.toa_sw_clr_c_mon
    ds = ds[["net","net_clr","lw","lw_clr","sw","sw_clr"]]
    ds["net_cre"] = ds.net - ds.net_clr
    ds["lw_cre"] = ds.lw - ds.lw_clr
    ds["sw_cre"] = ds.sw - ds.sw_clr

    # proper years and weighting
    return (
        ds.sel(time=slice("2000-03-01", "2026-03-01"))
        .pipe(add_weights)
        .pipe(march_to_feb_years)
        .pipe(seasonal_means)
    )

ds = read_ceres_raw()

# save linear trends
to_trend(ds).to_netcdf(f"pp/ceres_trends.nc")
print("done linear trend")

# save global-mean
ds_gm = global_mean(ds).load()
gm_trend = trend_and_ci(ds_gm)
gm_trend.to_netcdf("pp/ceres_gm_timeseries.nc")
ds_gm.close()
gm_trend.close()
print("done global-mean")