import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import t
import xarray as xr


###
# make regime masks
# (1986-2000 climatology; omega uses 1991-2000)
###

SIGMA_LAT = 1.5
SIGMA_LON = 1.5
TROPICAL_LAT = 40
POLAR_LAT = 60
SIC_THRESH = 0.1
LAND_THRESH = 0.1
OMEGA_THRESH = 0.0
NH_STORM_FACTOR = 0.15
SH_STORM_FACTOR = 0.15
CONFIDENCE_LEVEL = 0.9
HPA2_TO_PA2 = 10_000.0
MIN_UNIT_GAP_DECADES = 2.0


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


def normalize_slp_units(slp_var):
    spatial_dimensions = [dimension for dimension in slp_var.dims if dimension != "year"]
    yearly_scale = slp_var.max(dim=spatial_dimensions, skipna=True)
    scale_values = np.asarray(yearly_scale.values, dtype=float)
    ordered_log_scale = np.sort(np.log10(scale_values))
    gaps = np.diff(ordered_log_scale)
    if gaps.size == 0 or float(gaps.max()) < MIN_UNIT_GAP_DECADES:
        return slp_var
    gap_index = int(np.argmax(gaps))
    split_log_scale = (
        ordered_log_scale[gap_index] + ordered_log_scale[gap_index + 1]
    ) / 2.0
    hpa2_year_mask = np.log10(yearly_scale) < split_log_scale
    return xr.where(hpa2_year_mask, slp_var * HPA2_TO_PA2, slp_var)


def normalize_slp_latitude(slp_var):
    sine_squared = np.sin(np.deg2rad(slp_var.lat)) ** 2
    latitude_factor = xr.where(np.abs(slp_var.lat) <= 10, 1.0, sine_squared)
    return slp_var / latitude_factor


def regimes_from_clim(fin, fout):
    ds = xr.open_dataset("pp/"+fin)

    lsm = ds.lsm.mean("year")
    siconc_lower, _ = confidence_bounds(ds.siconc)
    omega500 = smooth_yearly(valid_years(ds.omega500))
    omega_lower, omega_upper = confidence_bounds(omega500)
    slp_var = normalize_slp_units(valid_years(ds.SLP_var))
    slp_var = smooth_yearly(normalize_slp_latitude(slp_var))
    slp_lower, _ = confidence_bounds(slp_var)
    slp_mean = slp_var.mean("year")

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

    # SLP_var_max for NH and SH
    slp_var_max_nh = slp_mean.where(masks.lat > 0).max(dim=("lat", "lon"))
    slp_var_max_sh = slp_mean.where(masks.lat < 0).max(dim=("lat", "lon"))
    masks["nh_storms"] = masks.nh_storms.where(
        (masks.lat > 0)
        & (slp_lower > NH_STORM_FACTOR * slp_var_max_nh)
        & (masks.nh_cryosphere < 0.5), 0
    )
    masks["sh_storms"] = masks.sh_storms.where(
        (masks.lat < 0)
        & (slp_lower > SH_STORM_FACTOR * slp_var_max_sh)
        & (masks.sh_cryosphere < 0.5), 0
    )

    masks["tropical_ascent"] = masks.tropical_ascent.where(
        (masks.lat >= -TROPICAL_LAT) & (masks.lat <= TROPICAL_LAT)
        & (omega_upper <= -OMEGA_THRESH)
        & (masks.nh_storms < 0.5) & (masks.sh_storms < 0.5), 0
    )
    masks["subsidence_land"] = masks.subsidence_land.where(
        (masks.lat >= -TROPICAL_LAT) & (masks.lat <= TROPICAL_LAT)
        & (omega_lower > OMEGA_THRESH)
        & (lsm >= LAND_THRESH)
        & (masks.nh_storms < 0.5) & (masks.sh_storms < 0.5), 0
    )
    masks["subsidence_ocean"] = masks.subsidence_ocean.where(
        (masks.lat >= -TROPICAL_LAT) & (masks.lat <= TROPICAL_LAT)
        & (omega_lower > OMEGA_THRESH)
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


regimes_from_clim("era5_clim.nc", "regime_masks.nc")
