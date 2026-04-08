import numpy as np
import xarray as xr
import xesmf as xe


trends = xr.open_mfdataset(["pp/ceres_trends.nc"]).drop_sel(season="ANN")
masks = xr.open_mfdataset(["pp/regime_masks.nc"])

seasons = trends.season.values
regimes = list(masks.keys())[:-1]
vars_ = list(trends.data_vars)  # all variables in dataset

# initialize empty containers for each variable
data = {
    v: xr.DataArray(
        dims=("season", "regime"),
        coords={"season": seasons, "regime": regimes},
    )
    for v in vars_
}

for s in seasons:
    da = trends.sel(season=s)
    area = masks.area.sel(season=s)

    for r in regimes:
        regime = masks[r].sel(season=s)

        regime_trend = (da * regime).weighted(area).mean(["lon", "lat"])

        for v in vars_:
            data[v].loc[dict(season=s, regime=r)] = regime_trend[v]

# combine into dataset
ds = xr.Dataset(data)

# add area of each regime to dataset
regimes = [r for r in masks.data_vars if r != "area"]
area = masks.area
# compute area per regime per season
area_da = xr.concat(
    [
        (masks[r] * area).sum(("lat", "lon")).assign_coords(regime=r)
        for r in regimes
    ],
    dim="regime"
)
ds["area"] = area_da
ds["area_fraction"] = area_da / ds.area.sum("regime")

# add annual mean
days_per_season = xr.DataArray(
    [90.65, 92, 92, 91],
    coords={"season": ["DJF", "MAM", "JJA", "SON"]},
    dims="season"
)
ann = ds.weighted(days_per_season).mean("season")
ann = ann.expand_dims(season=["ANN"])
ds = xr.concat([ds, ann], dim="season")

ds.to_netcdf("pp/regime_mean_trends.nc")
