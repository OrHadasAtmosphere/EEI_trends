import pandas as pd
import xarray as xr
import xesmf as xe

from . import add_weights

season_def = {
    "MAM":[3,4,5],
    "JJA":[6,7,8],
    "SON":[9,10,11],
    "DJF":[12,1,2],
}

def process_era5_clim():
    ceres = xr.open_dataset("pp/ceres_trends.nc")
    ceres = ceres.sortby(["lat","lon"])

    # read ERA5 climatology
    ds = xr.open_mfdataset(["raw_data/masks_levels.nc","raw_data/masks_pressures.nc"])
    ds = ds.rename({"valid_time":"time","latitude":"lat","longitude":"lon"})
    ds["omega500"] = ds.sel(pressure_level=500).w
    ds = ds.drop_vars(["pressure_level","number","expver","w"])
    wrapped = xr.concat([ds, ds.isel(lon=0)],dim="lon")
    ds = wrapped.assign_coords(lon=list(ds.lon.values) + [ds.lon.values[0] + 360])

    # read SLP variance climatology
    slp = xr.open_mfdataset(["raw_data/SLP_var_1990_2000.nc","raw_data/SLP_var_2001_2010.nc"])

    # make "year" and "month" into one "time" coord
    time = pd.to_datetime(
        [f"{y}-{m:02d}-01" for y in slp.year.values for m in slp.month.values]
    )
    slp = slp.stack(time=("year", "month"))
    slp = slp.drop_vars(["year","month","time"])
    slp = slp.assign_coords(time=time)

    # read storm data -- counts of storms within 1000km
    cyc = xr.open_dataset("raw_data/cyclone_anticyclone_1000km_fraction_1986_2024.nc").sel(time=slice("1990","2025"))[[
        "monthly_cyclone_day_fraction","monthly_anticyclone_day_fraction","monthly_number_of_days"]]
    cyc["monthly_storm_day_fraction"] = (cyc.monthly_cyclone_day_fraction + cyc.monthly_anticyclone_day_fraction).clip(max=1)
    cyc = cyc.drop_vars(["monthly_cyclone_day_fraction","monthly_anticyclone_day_fraction"])
    cyc = cyc.rename({"longitude":"lon","latitude":"lat"})
    wrapped = xr.concat([cyc, cyc.isel(lon=0)],dim="lon")
    cyc = wrapped.assign_coords(lon=list(cyc.lon.values) + [cyc.lon.values[0] + 360])

    # regrid to same 1x1 from ceres
    regridder = xe.Regridder(ds, ceres, method="bilinear")
    ds = regridder(ds)
    regridder = xe.Regridder(slp, ceres, method="bilinear")
    slp = regridder(slp)
    regridder = xe.Regridder(cyc, ceres, method="bilinear")
    cyc = regridder(cyc)
    
    ds = xr.merge([ds,slp,cyc])

    # add days-in-month weights
    return add_weights(ds)

# make seasonal clim
def seasonal_clim(ds, slice=slice("1990-03-01", "2000-03-01"), prefix="", outfile_ext=""):
    ds = ds.sel(time=slice)
    all = []
    for seas in season_def.keys():
        sub = ds.sel(time=ds.time.dt.month.isin(season_def[seas]))
        sub_mean = sub.weighted(sub.days_in_month).mean("time")
        sub_mean = sub_mean.expand_dims(season=[seas])
        all.append(sub_mean)
    ds = xr.concat(all, dim="season")
    ds = ds.drop_vars(["days_in_month"])
    ds.to_netcdf(f"pp/{prefix}era5_clim{outfile_ext}.nc")

if __name__ == "__main__":
    seasonal_clim(process_era5_clim())
