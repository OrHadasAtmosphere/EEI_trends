import numpy as np
import xarray as xr
from scipy import stats

def add_weights(ds):
    weights = ds.time.dt.days_in_month
    weights = weights.where(weights.time.dt.month!=2, 28.65)
    ds["days_in_month"] = weights
    return ds

def lat_mean(ds):
    return ds.weighted(np.cos(np.deg2rad(ds.lat))).mean("lat")

def global_mean(ds):
    return lat_mean(ds.mean("lon"))

def trend_and_ci(ds, dim="year", alpha=0.05):
    def linregress_1d(y, x):
        res = stats.linregress(x, y)
        fit = res.intercept + res.slope * x
        tcrit = stats.t.ppf(1 - alpha/2, len(x) - 2)
        ci = tcrit * res.stderr
        return fit, res.slope, ci

    y = ds[["net","net_clr","net_cre","sw","sw_clr","sw_cre","lw","lw_clr","lw_cre"]]
    fit, slope, ci = xr.apply_ufunc(
        linregress_1d,
        y, ds[dim],
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )
    slope = slope*10 # per decade
    ci = ci*10 # per decade
    
    fit = fit.rename({v: f"{v}_fit" for v in fit.data_vars})
    slope = slope.rename({v: f"{v}_slope_mean" for v in slope.data_vars})
    ci = ci.rename({v: f"{v}_slope_ci" for v in ci.data_vars})
    ds = xr.merge([ds,fit,slope,ci])
    return ds
