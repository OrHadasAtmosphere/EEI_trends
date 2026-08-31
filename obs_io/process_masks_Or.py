import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import t
import xarray as xr


###
# make regime masks
# union of 1986-2000 and 2011-2025 criteria; omega starts in 1991
###

SIGMA_LAT = 1.5
SIGMA_LON = 1.5
TROPICAL_LAT = 40
POLAR_LAT = 60
SIC_THRESH = 0.1
LAND_THRESH = 0.1
OMEGA_THRESH = 0.005
STORM_PROXIMITY_THRESHOLD = 0.2
STORM_PROXIMITY_FILE = (
    "raw_data/cyclone_anticyclone_1000km_fraction_1986_2024.nc"
)
CONFIDENCE_LEVEL = 0.9
EARLY_YEARS = range(1986, 2001)
EARLY_OMEGA_YEARS = range(1991, 2001)
LATE_YEARS = range(2016, 2026)
EARLY_STORM_YEARS = range(1986, 2001)
LATE_STORM_YEARS = range(2011, 2025)


def smooth(da):
    return gaussian_filter(
        da,
        sigma=(SIGMA_LAT, SIGMA_LON),
        mode=("nearest", "wrap")  # lat, lon
    )


def smooth_yearly(da):
    return xr.apply_ufunc(
        smooth,
        da,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[da.dtype],
    )


def valid_years(da):
    dimensions = [dimension for dimension in da.dims if dimension != "year"]
    year_has_data = da.notnull().any(dim=dimensions)
    return da.sel(year=da.year.where(year_has_data, drop=True))


def confidence_bounds(yearly):
    yearly = valid_years(yearly)
    sample_size = yearly.sizes["year"]
    if sample_size < 2:
        raise ValueError("at least two years are required for a confidence interval")
    mean = yearly.mean("year")
    variance = ((yearly - mean) ** 2).sum("year", skipna=True) / (sample_size - 1)
    standard_error = np.sqrt(variance / sample_size)
    critical_value = t.ppf(CONFIDENCE_LEVEL, df=sample_size - 1)
    return mean - critical_value * standard_error, mean + critical_value * standard_error


def period_data(da, years):
    return valid_years(da.sel(year=list(years)))


def storm_proximity_data(ds):
    variable_names = ["cyclone_day_fraction", "anticyclone_day_fraction"]
    if all(name in ds for name in variable_names):
        proximity = ds[variable_names]
    else:
        storm_years = [*EARLY_STORM_YEARS, *LATE_STORM_YEARS]
        with xr.open_dataset(STORM_PROXIMITY_FILE) as proximity_ds:
            proximity = proximity_ds[variable_names].rename(
                {"latitude": "lat", "longitude": "lon"}
            )
            proximity = proximity.sel(year=storm_years, season=ds.season)
            proximity = proximity.interp(lat=ds.lat, lon=ds.lon).load()
    return valid_years(
        proximity.cyclone_day_fraction + proximity.anticyclone_day_fraction
    )


def regimes_from_clim(fin, fout):
    ds = xr.open_dataset("pp/"+fin)

    lsm = period_data(ds.lsm, EARLY_YEARS).mean("year")
    siconc_lower, _ = confidence_bounds(period_data(ds.siconc, EARLY_YEARS))
    omega500 = smooth_yearly(valid_years(ds.omega500))
    early_omega_lower, early_omega_upper = confidence_bounds(
        period_data(omega500, EARLY_OMEGA_YEARS)
    )
    late_omega_lower, late_omega_upper = confidence_bounds(
        period_data(omega500, LATE_YEARS)
    )
    storm_proximity = storm_proximity_data(ds)
    early_storm_lower, _ = confidence_bounds(
        period_data(storm_proximity, EARLY_STORM_YEARS)
    )
    late_storm_lower, _ = confidence_bounds(
        period_data(storm_proximity, LATE_STORM_YEARS)
    )
    storm_criterion = (
        (early_storm_lower > STORM_PROXIMITY_THRESHOLD)
        | (late_storm_lower > STORM_PROXIMITY_THRESHOLD)
    )

    early_ascent = early_omega_upper <= -OMEGA_THRESH
    early_descent = early_omega_lower > OMEGA_THRESH
    late_ascent = late_omega_upper <= -OMEGA_THRESH
    late_descent = late_omega_lower > OMEGA_THRESH
    ascent_criterion = early_ascent | (~early_descent & late_ascent)
    descent_criterion = early_descent | (~early_ascent & late_descent)

    # define masks
    masks = xr.Dataset()
    for name in [
        "nh_cryosphere", "sh_cryosphere", "nh_storms", "sh_storms",
        "subsidence_land", "subsidence_ocean", "tropical_ascent", "residual"
    ]:
        masks[name] = xr.ones_like(lsm)

    masks["nh_cryosphere"] = masks.nh_cryosphere.where(
        ((masks.lat >= POLAR_LAT) & (siconc_lower > SIC_THRESH)) # sea ice >60
        | ((masks.lat >= 75) & (lsm > LAND_THRESH)) # any land >75
        | ((masks.lat >= POLAR_LAT) & (lsm > LAND_THRESH) & (masks.lon > 300) & (masks.lon < 350)), 0
    ) # greenland
    masks["sh_cryosphere"] = masks.sh_cryosphere.where(
        ((masks.lat <= -POLAR_LAT) & (siconc_lower > SIC_THRESH)) # sea ice >60
        | ((masks.lat <= -65) & (lsm > LAND_THRESH)), 0
    ) # any land >65

    masks["nh_storms"] = masks.nh_storms.where(
        (masks.lat > 0)
        & storm_criterion
        & (masks.nh_cryosphere < 0.5), 0
    )
    masks["sh_storms"] = masks.sh_storms.where(
        (masks.lat < 0)
        & storm_criterion
        & (masks.sh_cryosphere < 0.5), 0
    )

    masks["tropical_ascent"] = masks.tropical_ascent.where(
        (masks.lat >= -TROPICAL_LAT) & (masks.lat <= TROPICAL_LAT)
        & ascent_criterion
        & (masks.nh_storms < 0.5) & (masks.sh_storms < 0.5), 0
    )
    masks["subsidence_land"] = masks.subsidence_land.where(
        (masks.lat >= -TROPICAL_LAT) & (masks.lat <= TROPICAL_LAT)
        & descent_criterion
        & (lsm >= LAND_THRESH)
        & (masks.nh_storms < 0.5) & (masks.sh_storms < 0.5), 0
    )
    masks["subsidence_ocean"] = masks.subsidence_ocean.where(
        (masks.lat >= -TROPICAL_LAT) & (masks.lat <= TROPICAL_LAT)
        & descent_criterion
        & (lsm < LAND_THRESH)
        & (masks.nh_storms < 0.5) & (masks.sh_storms < 0.5), 0
    )

    masks["residual"] = masks.residual.where(
        (masks.nh_cryosphere < 0.5) & (masks.sh_cryosphere < 0.5)
        & (masks.nh_storms < 0.5) & (masks.sh_storms < 0.5)
        & (masks.subsidence_land < 0.5) & (masks.subsidence_ocean < 0.5)
        & (masks.tropical_ascent < 0.5), 0
    )

    # add grid-area
    R = 6371000  # radius of Earth / m
    dlat = np.deg2rad(float(masks.lat.diff("lat").mean()))
    dlon = np.deg2rad(float(masks.lon.diff("lon").mean()))
    area = (R**2) * dlat * dlon * np.cos(np.deg2rad(masks.lat))
    masks["area"] = area.broadcast_like(masks.nh_cryosphere)

    # check all points are in exactly one mask
    mask_sum = sum(masks[name] for name in masks.data_vars if name != "area")
    assert np.all(mask_sum == 1), "Not all values are 1"

    masks.to_netcdf("pp/"+fout)


if __name__ == "__main__":
    regimes_from_clim("era5_clim.nc", "regime_masks.nc")