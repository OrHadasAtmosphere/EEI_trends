import xarray as xr
from . import add_weights, process_era5_clim

season_def = {
    "MAM":[3,4,5],
    "JJA":[6,7,8],
    "SON":[9,10,11],
    "DJF":[12,1,2],
}

ds_full = process_era5_clim()

# make seasonal clim
def seasonal_clim(ds, slice=slice("1990-03-01", "2000-03-01"), outfile_ext=""):
    ds = ds_full.sel(time=slice)
    all = []
    for seas in season_def.keys():
        sub = ds.sel(time=ds.time.dt.month.isin(season_def[seas]))
        sub_mean = sub.weighted(sub.days_in_month).mean("time")
        sub_mean = sub_mean.expand_dims(season=[seas])
        all.append(sub_mean)
    ds = xr.concat(all, dim="season")
    ds = ds.drop_vars(["days_in_month"])
    ds.to_netcdf(f"pp/era5_clim{outfile_ext}.nc")

seasonal_clim(ds_full)
