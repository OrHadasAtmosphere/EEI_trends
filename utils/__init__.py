import numpy as np
import xarray as xr
from scipy import stats

def lat_mean(ds):
    return ds.weighted(np.cos(np.deg2rad(ds.lat))).mean("lat")

def global_mean(ds):
    return lat_mean(ds.mean("lon"))

def trend_and_ci(ds, dim="time", alpha=0.05, vars=None):
    
    if isinstance(vars, str):
        vars = [vars]
    
    if vars is None:
        vars = [v for v in ds.data_vars if ds[v].dtype in [np.float32, np.float64]]
    
    if dim not in ds.dims:
        for potential_dim in ['time', 'year', 'month']:
            if potential_dim in ds.dims:
                dim = potential_dim
                break
        else:
            raise ValueError(f"Dimension '{dim}' not found in dataset. Available: {list(ds.dims)}")
    
    def linregress_1d(y, x):
        res = stats.linregress(x, y)
        fit = res.intercept + res.slope * x
        tcrit = stats.t.ppf(1 - alpha/2, len(x) - 2)
        ci = tcrit * res.stderr
        return fit, res.slope, ci

    y = ds[vars]
    fit, slope, ci = xr.apply_ufunc(
        linregress_1d,
        y, ds[dim],
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )
    slope = slope*10  # per decade
    ci = ci*10  # per decade
    
    fit = fit.rename({v: f"{v}_fit" for v in fit.data_vars})
    slope = slope.rename({v: f"{v}_slope_mean" for v in slope.data_vars})
    ci = ci.rename({v: f"{v}_slope_ci" for v in ci.data_vars})
    ds = xr.merge([ds, fit, slope, ci])
    return ds
