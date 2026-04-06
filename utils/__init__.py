import numpy as np
from scipy import stats

def add_weights(ds):
    weights = ds.time.dt.days_in_month
    weights = weights.where(weights.time.dt.month!=2, 28.65)
    ds["days_in_month"] = weights
    return ds

def lat_mean(ds):
    return ds.weighted(np.cos(np.deg2rad(ds.lat))).mean("lat")

def global_mean(da):
    return lat_mean(da.mean("lon"))

def global_trend_and_ci(da):
    """returns global annual mean slope, fit, and 95% CI from data array"""
    gm = global_mean(da)
    result = stats.linregress(gm.year.values, gm.values)
    t_critical = stats.t.ppf(0.975, gm.year.size - 2)
    ci95 = t_critical * result.stderr
    
    fit = result.intercept + result.slope * gm.year.values 
    
    return fit, result.slope, ci95