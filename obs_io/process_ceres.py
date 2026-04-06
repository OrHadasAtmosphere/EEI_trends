import numpy as np
import xarray as xr

from utils import global_mean, add_weights, global_trend_and_ci

season_def = {
    "ANN":np.arange(12)+1,
    "MAM":[3,4,5],
    "JJA":[6,7,8],
    "SON":[9,10,11],
    "DJF":[12,1,2],
}

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

annual_net_toa = ds.sel(season='ANN').net
fit, slope, ci = global_trend_and_ci(annual_net_toa)

ds_gm = global_mean(ds)
ds_gm['annual_linear_fit'] = fit
ds_gm.assign_attrs({"decadal_net_trend": slope*10, "net_trend_ci": ci*10}).to_netcdf("pp/ceres_gm_timeseries.nc")

# calculate trends
ds_trend = ds.polyfit("year",1).sel(degree=1).drop_vars(["degree","days_in_month_polyfit_coefficients"])*10
ds_trend = ds_trend.rename({
    v: v.replace("_polyfit_coefficients", "")
    for v in ds_trend.data_vars
})
ds_trend.to_netcdf("pp/ceres_trends.nc")
