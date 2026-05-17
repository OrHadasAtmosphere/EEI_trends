import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr

from utils import trend_and_ci

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

def read_ceres(pathname="raw_data/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202602.nc"):
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

def regime_trend(ds, mask_file, outfile):
    da = ds.drop_sel(season="ANN").drop_vars(["days_in_month"])
    masks = xr.open_mfdataset(["pp/"+mask_file])
    regimes = [r for r in masks.data_vars if r != "area"]

    # average over regimes
    rm = xr.concat(
        [
            (da * masks[r])
            .weighted(masks.area)
            .mean(("lat", "lon"))
            .expand_dims(regime=[r])
            for r in regimes
        ],
        dim="regime"
    )

    # compute area per regime per season
    area_da = xr.concat(
        [
            (masks[r] * masks.area).sum(("lat", "lon")).assign_coords(regime=r)
            for r in regimes
        ],
        dim="regime"
    )
    rm["area"] = area_da
    rm["area_fraction"] = area_da / rm.area.sum("regime")

    # add annual mean
    days_per_season = xr.DataArray(
        [90.65, 92, 92, 91],
        coords={"season": ["DJF", "MAM", "JJA", "SON"]},
        dims="season"
    )
    ann = rm.weighted(days_per_season).mean("season")
    ann = ann.expand_dims(season=["ANN"])
    rm = xr.concat([rm, ann], dim="season")

    rm = rm.load()
    rm_trend = trend_and_ci(rm)
    rm_trend.to_netcdf("pp/"+outfile)
    print(f"done regime-mean: {outfile}")

def regimes_from_clim(fin, fout):
    ds = xr.open_dataset("pp/"+fin)
    
    # define masks
    ds["nh_cryosphere"] = (ds.lsm>-1)*1
    ds["sh_cryosphere"] = (ds.lsm>-1)*1
    ds["nh_storms"] = (ds.lsm>-1)*1
    ds["sh_storms"] = (ds.lsm>-1)*1
    ds["subsidence_land"] = (ds.lsm>-1)*1
    ds["subsidence_ocean"] = (ds.lsm>-1)*1
    ds["tropical_ascent"] = (ds.lsm>-1)*1
    ds["residual"] = (ds.lsm>-1)*1
    
    # some parameters
    SIGMA_LAT = 1.5
    SIGMA_LON = 1.5
    TROPICAL_LAT = 40
    POLAR_LAT = 60
    SIC_THRESH = 0.1
    LAND_THRESH = 0.1
    OMEGA_THRESH = 0.0
    NH_STORM_FACTOR = 0.2 # TODO
    SH_STORM_FACTOR = 0.3
    
    # smooth dataarray
    def smooth(da):
        return gaussian_filter(
            da,
            sigma=(SIGMA_LAT, SIGMA_LON),
            mode=("nearest", "wrap")  # lat, lon
        )
    
    ds["omega500"] = xr.apply_ufunc(
        smooth,
        ds.omega500,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ds.omega500.dtype],
    )
    ds["SLP_var"] = xr.apply_ufunc(
        smooth,
        ds.SLP_var,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[ds.SLP_var.dtype],
    )
    
    
    # define masks
    ds["nh_cryosphere"] = ds.nh_cryosphere.where(((ds.lat >= POLAR_LAT) & (ds.siconc > SIC_THRESH)) # sea ice >60
                                        | ((ds.lat >=75) & (ds.lsm > LAND_THRESH)) # any land >75 
                                        | ((ds.lat >=POLAR_LAT) & (ds.lsm > LAND_THRESH) & (ds.lon > 300) & (ds.lon < 350)), 0) # greenland
    ds["sh_cryosphere"] = ds.sh_cryosphere.where(((ds.lat <= -POLAR_LAT) & (ds.siconc > SIC_THRESH)) # sea ice >60
                                        | ((ds.lat <= -65) & (ds.lsm > LAND_THRESH)), 0) # any land >65
    
    # SLP_var_max for NH and SH
    ds["SLP_var_max_nh"] = ds.SLP_var.where(ds.lat > 0).max(dim=("lat", "lon"))
    ds["SLP_var_max_sh"] = ds.SLP_var.where(ds.lat < 0).max(dim=("lat", "lon"))
    ds["nh_storms"] = ds.nh_storms.where((ds.lat > 0) & (ds.SLP_var > NH_STORM_FACTOR*ds.SLP_var_max_nh) & (ds.nh_cryosphere < 0.5), 0)
    ds["sh_storms"] = ds.sh_storms.where((ds.lat < 0) & (ds.SLP_var > SH_STORM_FACTOR*ds.SLP_var_max_sh) & (ds.sh_cryosphere < 0.5), 0)
    
    ds["tropical_ascent"] = ds.tropical_ascent.where((ds.lat >= -TROPICAL_LAT) & (ds.lat <= TROPICAL_LAT) 
                                        & (ds.omega500 <= -OMEGA_THRESH) 
                                        & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5), 0)
    ds["subsidence_land"] = ds.subsidence_land.where((ds.lat >= -TROPICAL_LAT) & (ds.lat <= TROPICAL_LAT) 
                                        & (ds.omega500 > OMEGA_THRESH) 
                                        & (ds.lsm >= LAND_THRESH) & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5), 0)
    ds["subsidence_ocean"] = ds.subsidence_ocean.where((ds.lat >= -TROPICAL_LAT) & (ds.lat <= TROPICAL_LAT) 
                                        & (ds.omega500 > OMEGA_THRESH) 
                                        & (ds.lsm < LAND_THRESH) & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5), 0)
    
    ds["residual"] = ds.residual.where((ds.nh_cryosphere < 0.5) & (ds.sh_cryosphere < 0.5) 
                                        & (ds.nh_storms < 0.5) & (ds.sh_storms < 0.5)
                                        & (ds.subsidence_land < 0.5) & (ds.subsidence_ocean < 0.5)
                                        & (ds.tropical_ascent < 0.5), 0)
    
    masks = ds.drop_vars(["lsm","siconc","omega500","SLP_var","SLP_var_max_nh","SLP_var_max_sh"])
    
    # add grid-area
    R = 6371000  # radius of Earth / m
    dlat = np.deg2rad(float(masks.lat.diff("lat").mean()))
    dlon = np.deg2rad(float(masks.lon.diff("lon").mean()))
    area = (R**2) * dlat * dlon * np.cos(np.deg2rad(masks.lat))
    masks["area"] = area.broadcast_like(masks)
    
    # check all points are in exactly one mask
    sum = masks.tropical_ascent + masks.subsidence_land + masks.subsidence_ocean + masks.nh_storms + masks.sh_storms + masks.nh_cryosphere + masks.sh_cryosphere + masks.residual
    assert np.all(sum == 1), "Not all values are 1"
    
    masks.to_netcdf("pp/"+fout)