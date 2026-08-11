import xarray as xr
from utils import trend_and_ci
from .ceres import read_ceres_raw


def regime_trend(ds, mask_file, outfile):
    da = ds.drop_sel(season="ANN").drop_vars(["days_in_month"])
    masks = xr.open_dataset("pp/"+mask_file).load()
    regimes = [r for r in masks.data_vars if r != "area"]
    regime_masks = masks[regimes].to_array("regime")

    # average over regimes
    rm = (
        (da * regime_masks)
        .weighted(masks.area)
        .mean(("lat", "lon"))
        .transpose("regime", "season", "year")
    )

    # compute area per regime per season
    area_da = (regime_masks * masks.area).sum(("lat", "lon"))
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


if __name__ == "__main__":
    regime_trend(read_ceres_raw(), "regime_masks.nc", "regime_mean_trends.nc")
