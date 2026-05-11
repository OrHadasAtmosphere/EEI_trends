import numpy as np
import xarray as xr

def add_weights(ds):
    weights = ds.time.dt.days_in_month
    weights = weights.where(weights.time.dt.month!=2, 28.65)
    ds["days_in_month"] = weights
    return ds

def march_to_feb_years(ds):
    year_adj = ds.time.dt.year - (ds.time.dt.month == 1) - (ds.time.dt.month == 2) # define year Mar-Feb
    return ds.assign_coords(year=year_adj)

def seasonal_means(ds):
    season_def = {
        "ANN":np.arange(12)+1,
        "MAM":[3,4,5],
        "JJA":[6,7,8],
        "SON":[9,10,11],
        "DJF":[12,1,2],
    }
    all = []
    for seas in season_def.keys():
        sub = ds.sel(time=ds.time.dt.month.isin(season_def[seas]))
        sub_mean = (sub*sub.days_in_month).groupby("year").sum("time") / sub.days_in_month.groupby("year").sum("time")
        sub_mean = sub_mean.expand_dims(season=[seas])
        all.append(sub_mean)
    return xr.concat(all, dim="season")

def to_trend(ds):
    ds_trend = ds.polyfit("year",1).sel(degree=1).drop_vars(["degree","days_in_month_polyfit_coefficients"])*10
    ds_trend = ds_trend.rename({
        v: v.replace("_polyfit_coefficients", "")
        for v in ds_trend.data_vars
    })
    return ds_trend
