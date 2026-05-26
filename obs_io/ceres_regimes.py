import xarray as xr
from utils import trend_and_ci
from .ceres import read_ceres_raw

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

regime_trend(read_ceres_raw(), "regime_masks.nc", "regime_mean_trends.nc")
    