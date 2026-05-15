import os
import xarray as xr
from . import add_weights, march_to_feb_years, seasonal_means, to_trend
from utils import global_mean, trend_and_ci

SAVE_CERES_RAW = os.environ.get("SAVE_CERES_RAW", "False")
SAVE_CERES_RAW = SAVE_CERES_RAW == "True"

# read ceres
ds = xr.open_mfdataset(["raw_data/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202602.nc"])
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
ds = ds.sel(time=slice("2000-03-01", "2026-03-01"))
ds = add_weights(ds)
ds = march_to_feb_years(ds)
ds = seasonal_means(ds)
slices = [slice(None, None), slice("2005", None), slice(None, "2021")]

if SAVE_CERES_RAW:
    trendfiles = ["ceres_trends.nc", "diff_yrs/ceres_trends_from_2005.nc", "diff_yrs/ceres_trends_until_2021.nc"]
    for slc, tfile in zip(slices, trendfiles):
        # save linear trends
        to_trend(ds.sel(year=slc)).to_netcdf(f"pp/{tfile}")
        print(f"done linear trend: with CERES record cut using subselection {slc.start}-{slc.stop}")

    # save global-mean
    ds_gm = global_mean(ds).load()
    gm_trend = trend_and_ci(ds_gm)
    gm_trend.to_netcdf("pp/ceres_gm_timeseries.nc")
    ds_gm.close()
    gm_trend.close()
    print("done global-mean")
else:
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

    fname_ext_per_slice = [".nc", "_using_ceres_from_2005.nc", "_using_ceres_until_2021.nc"]
    for slc, nm_ext in zip(slices, fname_ext_per_slice):
        parent_dir = "diff_yrs/" if nm_ext != ".nc" else ""
        print(f"calculating regime trends for sliced CERES years: {slc.start}-{slc.stop}")
        regime_trend(ds.sel(year=slc), "regime_masks.nc", f"{parent_dir}regime_mean_trends{nm_ext}")
    
    print("\ncalculating regime-mean trends using different climatologies to define regimes:")

    for fmask, fout in zip(
        ["diff_yrs/regime_masks_1995.nc", "diff_yrs/regime_masks_2000.nc"],
        ["diff_yrs/regime_mean_trends_1995.nc", "diff_yrs/regime_mean_trends_2000.nc"]
    ):
        regime_trend(ds, fmask, fout)
        