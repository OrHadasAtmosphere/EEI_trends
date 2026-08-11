import xarray as xr
import xesmf as xe

from . import add_weights

SEASON_DEF = {
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "DJF": [12, 1, 2],
}
LEVEL_YEARS = range(1986, 2010)
OMEGA_YEARS = range(1991, 2026)
EARLY_YEARS = range(1986, 2001)
LATE_YEARS = range(2011, 2026)
MASK_YEARS = [*EARLY_YEARS, *LATE_YEARS]


def seasonal_yearly(ds, years):
    """Return complete, day-weighted seasonal means for each requested year."""
    season_year = xr.where(ds.time.dt.month == 12, ds.time.dt.year, ds.time.dt.year - 1)
    all_years = []
    for year in years:
        all_seasons = []
        for season, months in SEASON_DEF.items():
            if season == "DJF":
                selected = ds.where(
                    ds.time.dt.month.isin(months) & (season_year == year), drop=True
                )
            else:
                selected = ds.where(
                    ds.time.dt.month.isin(months) & (ds.time.dt.year == year), drop=True
                )
            if selected.sizes["time"] != 3:
                raise ValueError(f"{season} {year} does not contain three months")
            seasonal_mean = selected.weighted(selected.days_in_month).mean("time")
            all_seasons.append(seasonal_mean.expand_dims(season=[season]))
        all_years.append(xr.concat(all_seasons, dim="season").expand_dims(year=[year]))
    return xr.concat(all_years, dim="year").drop_vars("days_in_month", errors="ignore")


def process_era5_clim():
    ceres = xr.open_dataset("pp/ceres_trends.nc").sortby(["lat", "lon"])

    levels = xr.open_dataset("raw_data/masks_levels.nc")
    levels = levels.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
    levels = levels.drop_vars(["number", "expver"], errors="ignore")
    levels = seasonal_yearly(add_weights(levels), LEVEL_YEARS)

    omega = xr.open_dataset("raw_data/ERA5_omega500_monthly_1991_2026-02.nc")
    omega = omega.rename({"latitude": "lat", "longitude": "lon"})
    omega["omega500"] = omega.sel(pressure_level=500).w
    omega = omega.drop_vars(["pressure_level", "number", "expver", "w"], errors="ignore")
    omega = seasonal_yearly(add_weights(omega), OMEGA_YEARS)

    proximity = xr.open_dataset(
        "raw_data/cyclone_anticyclone_1000km_fraction_1986_2024.nc"
    )[["cyclone_day_fraction", "anticyclone_day_fraction"]]
    proximity = proximity.rename({"latitude": "lat", "longitude": "lon"})

    regridder = xe.Regridder(levels, ceres, method="bilinear")
    levels = regridder(levels)
    regridder = xe.Regridder(omega, ceres, method="bilinear")
    omega = regridder(omega)
    regridder = xe.Regridder(proximity, ceres, method="bilinear")
    proximity = regridder(proximity)

    return xr.merge([levels, omega, proximity], join="outer").sel(
        season=list(SEASON_DEF)
    )


def seasonal_clim(ds, time_slice=MASK_YEARS, outfile_ext=""):
    if isinstance(time_slice, slice):
        start = int(str(time_slice.start)[:4]) if time_slice.start is not None else int(ds.year.min())
        stop = int(str(time_slice.stop)[:4]) - 1 if isinstance(time_slice.stop, str) else time_slice.stop
        stop = int(ds.year.max()) if stop is None else int(stop)
        ds = ds.sel(year=slice(start, stop))
    elif time_slice is not None:
        ds = ds.sel(year=list(time_slice))
    encoding = {
        variable: {"zlib": True, "complevel": 4} for variable in ds.data_vars
    }
    ds.to_netcdf(f"pp/era5_clim{outfile_ext}.nc", encoding=encoding)


seasonal_clim(process_era5_clim())
