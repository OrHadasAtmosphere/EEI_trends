import xarray as xr
from . import march_to_feb_years, add_weights, seasonal_means, to_trend
from utils import global_mean, global_trend_and_ci

# read ceres
ds = xr.open_mfdataset(["raw_data/CERES_EBAF-TOA_Edition4.2.1_200003-202601.nc"])
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
ds = march_to_feb_years(ds)
ds = add_weights(ds)
ds = seasonal_means(ds)

annual_net_toa = ds.sel(season='ANN').net
fit, slope, ci = global_trend_and_ci(annual_net_toa)

ds_gm = global_mean(ds)
ds_gm['annual_linear_fit'] = fit
ds_gm.assign_attrs({"decadal_net_trend": slope*10, "net_trend_ci": ci*10}).to_netcdf("pp/ceres_gm_timeseries.nc")

to_trend(ds).to_netcdf("pp/ceres_trends.nc")
