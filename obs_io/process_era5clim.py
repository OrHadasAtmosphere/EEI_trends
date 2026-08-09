import xarray as xr
import xesmf as xe

from . import add_weights

SEASON_DEF = {
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "DJF": [12, 1, 2],
}
DATA_YEARS = range(1986, 2010)
OMEGA_YEARS = range(1991, 2010)
CONTROL_YEARS = slice(1986, 2000)


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
    levels = seasonal_yearly(add_weights(levels), DATA_YEARS)

    omega = xr.open_dataset("raw_data/ERA5_omega500_monthly_1991_2026-02.nc")
    omega = omega.rename({"latitude": "lat", "longitude": "lon"})
    omega["omega500"] = omega.sel(pressure_level=500).w
    omega = omega.drop_vars(["pressure_level", "number", "expver", "w"], errors="ignore")
    omega = seasonal_yearly(add_weights(omega), OMEGA_YEARS)

    slp = xr.open_dataset("raw_data/SLP_var_2_10day_1940_2025.nc")
    slp = slp.rename({"latitude": "lat", "longitude": "lon"})
    slp = slp.sel(year=list(DATA_YEARS))

    regridder = xe.Regridder(levels, ceres, method="bilinear")
    levels = regridder(levels)
    regridder = xe.Regridder(omega, ceres, method="bilinear")
    omega = regridder(omega)
    regridder = xe.Regridder(slp, ceres, method="bilinear")
    slp = regridder(slp)

    return xr.merge([levels, omega, slp], join="outer").sel(
        season=list(SEASON_DEF)
    )


def seasonal_clim(ds, time_slice=CONTROL_YEARS, outfile_ext=""):
    if time_slice is not None:
        start = int(str(time_slice.start)[:4]) if time_slice.start is not None else int(ds.year.min())
        stop = int(str(time_slice.stop)[:4]) - 1 if isinstance(time_slice.stop, str) else time_slice.stop
        stop = int(ds.year.max()) if stop is None else int(stop)
        ds = ds.sel(year=slice(start, stop))
    encoding = {
        variable: {"zlib": True, "complevel": 4} for variable in ds.data_vars
    }
    ds.to_netcdf(f"pp/era5_clim{outfile_ext}.nc", encoding=encoding)


seasonal_clim(process_era5_clim())
