from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.stats as stats
import xarray as xr

from trend_utils import fit_trend_map


STORM_MATRICS = "SLP"
CENTRAL_LONGITUDE = -135
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
FIGURES_DIR = SCRIPT_DIR / "figures"
OVERLEAF_FIGURES_DIR = Path(
    "/Users/orhadas/Library/CloudStorage/Dropbox-WeizmannInstitute/Or Hadas/Apps/Overleaf/Regional_EEI/figures"
)
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

CERES_INPUT_FILE = Path(
    "/Users/orhadas/Documents/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202512.nc"
)
CERES_OUTPUT_FILE = OUTPUT_DIR / "manuscript_ceres_diagnostics.nc"
MASKS_OUTPUT_FILE = OUTPUT_DIR / "manuscript_regime_masks.nc"
CONTRIBUTIONS_OUTPUT_FILE = OUTPUT_DIR / "manuscript_regime_contributions.nc"
SST_INPUT_FILE = OUTPUT_DIR / "SST_raw.nc"
SST_OUTPUT_FILE = OUTPUT_DIR / "manuscript_sst_diagnostics.nc"
OMEGA_FILE = OUTPUT_DIR / "W_mean.nc"
ICE_FILE = OUTPUT_DIR / "ice_mean.nc"
STORM_FILE = OUTPUT_DIR / f"{STORM_MATRICS}_var_yearly_OnlyTime.nc"
INSOLATION_FILE = OUTPUT_DIR / "sun_insolation.nc"

SEASONS = ("DJF", "MAM", "JJA", "SON")
PERIODS = ("annual", *SEASONS)
SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
SKY_LABELS = {
    "all_sky": "all-sky",
    "clear_sky": "clear-sky",
    "all_clear": "all-sky minus clear-sky",
}
REGION_NAMES = (
    "NH Polar Cryosphere",
    "SH Polar Cryosphere",
    "NH Storm Track",
    "SH Storm Track",
    "Tropical Descent",
    "Deserts",
    "Tropical Ascent",
    "Remaider",
)

STORM_TRACK_NH_FACTOR = 0.3
STORM_TRACK_SH_FACTOR = 0.4
OMEGA_POS_FACTOR = 0.05
OMEGA_NEG_FACTOR = 0.05
ICE_FACTOR = 0.1
OMEGA_LATITUDE_LIMIT = 40.0
RADIATION_TREND_START_YEAR = 2001
RADIATION_TREND_END_YEAR = 2024
POLAR_LATITUDE_LIMIT = 60.0


def ensure_required_variables(ds: xr.Dataset, required: set[str], context: str) -> None:
    missing = sorted(required.difference(ds.data_vars))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{context} is missing required variables: {joined}")


def figure_output_paths(filename: str) -> tuple[Path, Path]:
    return FIGURES_DIR / filename, OVERLEAF_FIGURES_DIR / filename


def save_figure_outputs(fig, filename: str, *, dpi: int = 300) -> tuple[Path, Path]:
    local_path, overleaf_path = figure_output_paths(filename)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    overleaf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(local_path, dpi=dpi)
    fig.savefig(overleaf_path, dpi=dpi)
    return local_path, overleaf_path


def save_text_outputs(
    text: str, local_path: Path, overleaf_path: Path
) -> tuple[Path, Path]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    overleaf_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(text)
    overleaf_path.write_text(text)
    return local_path, overleaf_path


def group_complete_means(
    da: xr.DataArray,
    labels: xr.DataArray,
    expected_count: int,
    output_dim: str,
    *,
    time_weights: xr.DataArray | None = None,
) -> xr.DataArray:
    if time_weights is None:
        grouped = da.groupby(labels).mean("time", skipna=True)
    else:
        weighted_sum = (da * time_weights).groupby(labels).sum("time", skipna=True)
        weight_sum = (
            time_weights.broadcast_like(da)
            .where(np.isfinite(da))
            .groupby(labels)
            .sum("time", skipna=True)
        )
        grouped = weighted_sum / weight_sum
    grouped = grouped.rename({labels.name: output_dim})

    counts = labels.groupby(labels).count()
    counts = counts.rename({labels.name: output_dim})
    valid_groups = counts.where(counts == expected_count, drop=True)[output_dim]
    return grouped.sel({output_dim: valid_groups})


def annual_year_labels(
    da: xr.DataArray,
    *,
    label_mode: str = "calendar",
) -> xr.DataArray:
    year = da["time"].dt.year.values.astype(np.int32)
    if label_mode == "march_to_february":
        year = year.copy()
        year[da["time"].dt.month.values <= 2] -= 1
    elif label_mode != "calendar":
        raise ValueError(f"Unsupported annual label mode: {label_mode}")

    return xr.DataArray(
        year,
        coords={"time": da["time"]},
        dims=("time",),
        name="year",
    )


def season_year_labels(
    da: xr.DataArray,
    season: str,
    *,
    djf_mode: str = "next_year",
) -> xr.DataArray:
    season_year = da["time"].dt.year.values.astype(np.int32)
    if season == "DJF":
        december = da["time"].dt.month.values == 12
        if djf_mode == "next_year":
            season_year = season_year + december.astype(np.int32)
        elif djf_mode == "december_year":
            season_year = season_year - (~december).astype(np.int32)
        else:
            raise ValueError(f"Unsupported DJF aggregation mode: {djf_mode}")

    return xr.DataArray(
        season_year,
        coords={"time": da["time"]},
        dims=("time",),
        name="season_year",
    )


def select_year_bounds(
    grouped: xr.DataArray,
    year_bounds: tuple[int, int] | None,
) -> xr.DataArray:
    if year_bounds is None:
        return grouped
    start_year, end_year = year_bounds
    return grouped.sel(year=slice(start_year, end_year))


def annual_means(
    da: xr.DataArray,
    *,
    year_bounds: tuple[int, int] | None = None,
    label_mode: str = "calendar",
    time_weights: xr.DataArray | None = None,
) -> xr.DataArray:
    labels = annual_year_labels(da, label_mode=label_mode)
    grouped = group_complete_means(
        da,
        labels,
        expected_count=12,
        output_dim="year",
        time_weights=time_weights,
    )
    return select_year_bounds(grouped, year_bounds)


def seasonal_means(
    da: xr.DataArray,
    season: str,
    *,
    year_bounds: tuple[int, int] | None = None,
    djf_mode: str = "next_year",
    time_weights: xr.DataArray | None = None,
) -> xr.DataArray:
    season_da = da.where(da["time"].dt.season == season, drop=True)
    season_weights = (
        None if time_weights is None else time_weights.sel(time=season_da["time"])
    )
    labels = season_year_labels(season_da, season, djf_mode=djf_mode)
    grouped = group_complete_means(
        season_da,
        labels,
        expected_count=3,
        output_dim="year",
        time_weights=season_weights,
    )
    return select_year_bounds(grouped, year_bounds)


def month_length_weights(da: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(
        da["time"].dt.days_in_month.values.astype(np.float64),
        coords={"time": da["time"]},
        dims=("time",),
        name="days_in_month",
    )


def seasonal_time_fractions(
    time: xr.DataArray,
    years: np.ndarray,
    *,
    djf_mode: str = "december_year",
) -> xr.DataArray:
    month_lengths = xr.DataArray(
        time.dt.days_in_month.values.astype(np.float64),
        coords={"time": time},
        dims=("time",),
        name="days_in_month",
    )
    season_weights: list[xr.DataArray] = []
    for season in SEASONS:
        season_lengths = month_lengths.where(time.dt.season == season, drop=True)
        labels = season_year_labels(season_lengths, season, djf_mode=djf_mode)
        grouped_days = season_lengths.groupby(labels).sum("time", skipna=True)
        grouped_days = grouped_days.rename({labels.name: "year"}).sel(year=years)
        season_weights.append(grouped_days.sum("year").expand_dims(season=[season]))

    total_days = xr.concat(season_weights, dim="season")
    return total_days / total_days.sum("season")


def area_weighted_global_mean(da: xr.DataArray) -> xr.DataArray:
    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(da["lat"].values)),
        coords={"lat": da["lat"]},
        dims=("lat",),
    )
    weights_2d = lat_weights.broadcast_like(da.isel(time=0, drop=True))
    valid_weights = weights_2d.where(np.isfinite(da))
    numerator = (da * valid_weights).sum(("lat", "lon"), skipna=True)
    denominator = valid_weights.sum(("lat", "lon"), skipna=True)
    return numerator / denominator


def linear_trend_series(
    series: xr.DataArray,
    coord: str = "year",
) -> tuple[float, xr.DataArray]:
    x = series[coord].values.astype(np.float64)
    y = series.values.astype(np.float64)
    valid = np.isfinite(y)
    if valid.sum() < 2:
        fit = xr.full_like(series, np.nan, dtype=np.float64)
        return np.nan, fit

    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    fit = xr.DataArray(
        intercept + slope * x,
        coords=series.coords,
        dims=series.dims,
    )
    return float(slope), fit


def build_period_statistics(
    da: xr.DataArray,
    *,
    year_bounds: tuple[int, int] | None = None,
    djf_mode: str = "next_year",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    means: list[xr.DataArray] = []
    trends: list[xr.DataArray] = []
    sample_counts: list[int] = []
    start_years: list[int] = []
    end_years: list[int] = []

    for period in PERIODS:
        grouped = (
            annual_means(da, year_bounds=year_bounds)
            if period == "annual"
            else seasonal_means(
                da,
                period,
                year_bounds=year_bounds,
                djf_mode=djf_mode,
            )
        )
        means.append(grouped.mean("year", skipna=True).expand_dims(period=[period]))
        trends.append(
            fit_trend_map(grouped, time_dim="year")["slope"].expand_dims(
                period=[period]
            )
        )
        sample_counts.append(int(grouped.sizes["year"]))
        start_years.append(int(grouped["year"].min()))
        end_years.append(int(grouped["year"].max()))

    coords = {"period": list(PERIODS)}
    return (
        xr.concat(means, dim="period").astype(np.float32),
        xr.concat(trends, dim="period").astype(np.float32),
        xr.DataArray(
            np.asarray(sample_counts, dtype=np.int16),
            coords=coords,
            dims=("period",),
            attrs={
                "long_name": "Number of complete means used in each fit",
                "units": "1",
            },
        ),
        xr.DataArray(
            np.asarray(start_years, dtype=np.int16),
            coords=coords,
            dims=("period",),
            attrs={"long_name": "First year used in each fit"},
        ),
        xr.DataArray(
            np.asarray(end_years, dtype=np.int16),
            coords=coords,
            dims=("period",),
            attrs={"long_name": "Last year used in each fit"},
        ),
    )


def load_aligned_ceres_fluxes(
    input_file: Path = CERES_INPUT_FILE,
    insolation_file: Path = INSOLATION_FILE,
) -> xr.Dataset:
    ds = xr.open_dataset(input_file, engine="netcdf4")
    insolation_ds = xr.open_dataset(insolation_file, engine="netcdf4")
    ensure_required_variables(
        ds,
        {
            "toa_sw_all_mon",
            "toa_lw_all_mon",
            "toa_sw_clr_c_mon",
            "toa_lw_clr_c_mon",
        },
        "CERES input",
    )
    ensure_required_variables(insolation_ds, {"solar_mon"}, "Insolation input")

    solar, sw_all, lw_all, sw_clear, lw_clear = xr.align(
        insolation_ds["solar_mon"],
        ds["toa_sw_all_mon"],
        ds["toa_lw_all_mon"],
        ds["toa_sw_clr_c_mon"],
        ds["toa_lw_clr_c_mon"],
        join="inner",
    )
    monthly = xr.Dataset(
        data_vars={
            "incoming_shortwave": solar.astype(np.float32),
            "all_sky_shortwave": sw_all.astype(np.float32),
            "all_sky_longwave": lw_all.astype(np.float32),
            "clear_sky_shortwave": sw_clear.astype(np.float32),
            "clear_sky_longwave": lw_clear.astype(np.float32),
        }
    )
    monthly["all_sky_net"] = (
        monthly["incoming_shortwave"]
        - monthly["all_sky_shortwave"]
        - monthly["all_sky_longwave"]
    ).astype(np.float32)
    monthly["clear_sky_net"] = (
        monthly["incoming_shortwave"]
        - monthly["clear_sky_shortwave"]
        - monthly["clear_sky_longwave"]
    ).astype(np.float32)
    monthly["all_clear_shortwave"] = (
        monthly["all_sky_shortwave"] - monthly["clear_sky_shortwave"]
    ).astype(np.float32)
    monthly["all_clear_longwave"] = (
        monthly["all_sky_longwave"] - monthly["clear_sky_longwave"]
    ).astype(np.float32)
    monthly["all_clear_net"] = (
        monthly["all_sky_net"] - monthly["clear_sky_net"]
    ).astype(np.float32)
    monthly.attrs.update(
        source_ceres_file=str(input_file),
        source_insolation_file=str(insolation_file),
    )
    return monthly


def fit_trend_with_confidence_interval(
    series: xr.DataArray,
    coord: str = "year",
) -> tuple[float, float, xr.DataArray, int]:
    x = series[coord].values.astype(np.float64)
    y = series.values.astype(np.float64)
    valid = np.isfinite(y)
    n_valid = int(valid.sum())
    if n_valid < 3:
        return np.nan, np.nan, xr.full_like(series, np.nan, dtype=np.float64), n_valid

    result = stats.linregress(x[valid], y[valid])
    t_critical = float(stats.t.ppf(0.975, n_valid - 2))
    ci95 = t_critical * float(result.stderr)
    fit = xr.DataArray(
        result.intercept + result.slope * x,
        coords=series.coords,
        dims=series.dims,
    )
    return float(result.slope), float(ci95), fit, n_valid


def area_weighted_mean_over_mask(
    da: xr.DataArray,
    mask: xr.DataArray,
) -> xr.DataArray:
    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(da["lat"].values)),
        coords={"lat": da["lat"]},
        dims=("lat",),
    ).broadcast_like(da.isel(year=0, drop=True))
    mask_on_grid = mask.sel(lat=da["lat"], lon=da["lon"])
    masked_weights = lat_weights.where(mask_on_grid)
    numerator = (da * masked_weights).sum(("lat", "lon"), skipna=True)
    denominator = masked_weights.where(np.isfinite(da)).sum(("lat", "lon"), skipna=True)
    return numerator / denominator


def effective_area_fraction(mask: xr.DataArray) -> float:
    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(mask["lat"].values)),
        coords={"lat": mask["lat"]},
        dims=("lat",),
    ).broadcast_like(mask)
    numerator = float(lat_weights.where(mask, 0.0).sum(skipna=True))
    denominator = float(lat_weights.sum(skipna=True))
    if np.isclose(denominator, 0.0):
        return np.nan
    return numerator / denominator


def build_ceres_dataset(input_file: Path = CERES_INPUT_FILE) -> xr.Dataset:
    monthly = load_aligned_ceres_fluxes(
        input_file=input_file, insolation_file=INSOLATION_FILE
    )
    time_weights = month_length_weights(monthly["all_sky_net"])
    radiation_year_bounds = (RADIATION_TREND_START_YEAR, RADIATION_TREND_END_YEAR)
    fields = {
        "all_sky": {
            "shortwave": monthly["all_sky_shortwave"],
            "longwave": monthly["all_sky_longwave"],
            "net": monthly["all_sky_net"],
        },
        "clear_sky": {
            "shortwave": monthly["clear_sky_shortwave"],
            "longwave": monthly["clear_sky_longwave"],
            "net": monthly["clear_sky_net"],
        },
        "all_clear": {
            "shortwave": monthly["all_clear_shortwave"],
            "longwave": monthly["all_clear_longwave"],
            "net": monthly["all_clear_net"],
        },
    }
    out = xr.Dataset(
        coords={
            "period": list(PERIODS),
            "lat": monthly["incoming_shortwave"]["lat"],
            "lon": monthly["incoming_shortwave"]["lon"],
        }
    )
    sample_count = start_year = end_year = None

    for sky_name, components in fields.items():
        for component_name, data in components.items():
            means, trends, counts, starts, ends = build_period_statistics(
                data,
                year_bounds=radiation_year_bounds,
                djf_mode="december_year",
            )
            if sample_count is None:
                sample_count = counts
                start_year = starts
                end_year = ends

            mean_name = f"{sky_name}_{component_name}_mean"
            trend_name = f"{sky_name}_{component_name}_trend"
            description_root = f"{SKY_LABELS[sky_name]} {component_name}"

            if component_name != "net":
                means.attrs.update(
                    long_name=f"{description_root} climatological mean",
                    units="W m-2",
                )
                out[mean_name] = means
            trends.attrs.update(
                long_name=f"{description_root} trend",
                units="W m-2 yr-1",
            )
            out[trend_name] = trends

    solar_means, solar_trends, _, _, _ = build_period_statistics(
        monthly["incoming_shortwave"],
        year_bounds=radiation_year_bounds,
        djf_mode="december_year",
    )
    solar_means.attrs.update(
        long_name="Incoming solar flux climatological mean",
        units="W m-2",
    )
    solar_trends.attrs.update(
        long_name="Incoming solar flux trend",
        units="W m-2 yr-1",
    )
    out["incoming_shortwave_mean"] = solar_means
    out["incoming_shortwave_trend"] = solar_trends

    out["all_sky_net_mean"] = (
        out["incoming_shortwave_mean"]
        - out["all_sky_shortwave_mean"]
        - out["all_sky_longwave_mean"]
    ).astype(np.float32)
    out["all_sky_net_mean"].attrs.update(
        long_name="all-sky net EEI climatological mean",
        units="W m-2",
        description="Incoming shortwave minus outgoing shortwave minus outgoing longwave.",
    )
    out["clear_sky_net_mean"] = (
        out["incoming_shortwave_mean"]
        - out["clear_sky_shortwave_mean"]
        - out["clear_sky_longwave_mean"]
    ).astype(np.float32)
    out["clear_sky_net_mean"].attrs.update(
        long_name="clear-sky net EEI climatological mean",
        units="W m-2",
        description="Incoming shortwave minus outgoing shortwave minus outgoing longwave.",
    )
    out["all_clear_net_mean"] = (
        out["all_sky_net_mean"] - out["clear_sky_net_mean"]
    ).astype(np.float32)
    out["all_clear_net_mean"].attrs.update(
        long_name="all-sky minus clear-sky net EEI climatological mean",
        units="W m-2",
    )

    out["all_sky_net_trend"] = (
        out["incoming_shortwave_trend"]
        - out["all_sky_shortwave_trend"]
        - out["all_sky_longwave_trend"]
    ).astype(np.float32)
    out["all_sky_net_trend"].attrs.update(
        long_name="all-sky net EEI trend",
        units="W m-2 yr-1",
        description="Incoming shortwave trend minus outgoing shortwave trend minus outgoing longwave trend.",
    )
    out["clear_sky_net_trend"] = (
        out["incoming_shortwave_trend"]
        - out["clear_sky_shortwave_trend"]
        - out["clear_sky_longwave_trend"]
    ).astype(np.float32)
    out["clear_sky_net_trend"].attrs.update(
        long_name="clear-sky net EEI trend",
        units="W m-2 yr-1",
        description="Incoming shortwave trend minus outgoing shortwave trend minus outgoing longwave trend.",
    )
    out["all_clear_net_trend"] = (
        out["all_sky_net_trend"] - out["clear_sky_net_trend"]
    ).astype(np.float32)
    out["all_clear_net_trend"].attrs.update(
        long_name="all-sky minus clear-sky net EEI trend",
        units="W m-2 yr-1",
    )

    global_eei = area_weighted_global_mean(monthly["all_sky_net"]).rename(
        "all_sky_global_net_monthly_mean"
    )
    global_eei.attrs.update(
        long_name="Global-mean all-sky EEI",
        units="W m-2",
        description=(
            "Computed as global-mean incoming shortwave minus global-mean outgoing "
            "shortwave minus global-mean outgoing longwave."
        ),
    )
    seasonal_global_eei = {
        season: seasonal_means(
            global_eei,
            season,
            year_bounds=radiation_year_bounds,
            djf_mode="december_year",
            time_weights=time_weights,
        )
        for season in SEASONS
    }
    common_years = seasonal_global_eei["DJF"]["year"].values.astype(np.int32)
    for season in SEASONS[1:]:
        common_years = np.intersect1d(
            common_years,
            seasonal_global_eei[season]["year"].values.astype(np.int32),
        )
    season_weights = seasonal_time_fractions(
        global_eei["time"], common_years, djf_mode="december_year"
    )
    annual_global_eei = xr.DataArray(
        np.tensordot(
            season_weights.values.astype(np.float64),
            np.stack(
                [
                    seasonal_global_eei[season].sel(year=common_years).values
                    for season in SEASONS
                ],
                axis=0,
            ),
            axes=(0, 0),
        ),
        coords={"year": common_years},
        dims=("year",),
        name="all_sky_global_net_annual_mean",
    ).astype(np.float32)
    annual_global_eei.name = "all_sky_global_net_annual_mean"
    annual_global_eei.attrs.update(
        long_name="Annual global-mean all-sky EEI",
        units="W m-2",
        description=(
            "Reconstructed from day-weighted seasonal global means using the same "
            "season-year labeling as the DJF diagnostics."
        ),
    )
    # fit trend and 95% confidence interval for the annual global series
    global_trend, global_ci95, annual_global_fit, _ = (
        fit_trend_with_confidence_interval(annual_global_eei, coord="year")
    )
    annual_global_fit = annual_global_fit.astype(np.float32)
    annual_global_fit.name = "all_sky_global_net_annual_fit"
    annual_global_fit.attrs.update(
        long_name="Linear fit to annual global-mean all-sky EEI",
        units="W m-2",
    )
    # store 95% CI for the slope (units: W m-2 yr-1)
    out["all_sky_global_net_annual_ci95"] = xr.DataArray(
        np.float32(global_ci95),
        attrs={
            "long_name": "95% confidence interval for trend in annual global-mean all-sky EEI",
            "units": "W m-2 yr-1",
        },
    )

    out["all_sky_global_net_monthly_mean"] = global_eei.astype(np.float32)
    out["all_sky_global_net_annual_mean"] = annual_global_eei
    out["all_sky_global_net_annual_fit"] = annual_global_fit
    out["all_sky_global_net_annual_trend"] = xr.DataArray(
        np.float32(global_trend),
        attrs={
            "long_name": "Trend in annual global-mean all-sky EEI",
            "units": "W m-2 yr-1",
        },
    )

    out["sample_count"] = sample_count
    out["start_year"] = start_year
    out["end_year"] = end_year
    out.attrs.update(
        title="Manuscript CERES diagnostics",
        source_file=str(input_file),
        source_insolation_file=str(INSOLATION_FILE),
        history="Created by basic_trend/calculate_manuscript_data.py",
        radiation_trend_year_range=(
            f"{RADIATION_TREND_START_YEAR}-{RADIATION_TREND_END_YEAR}"
        ),
        djf_definition=(
            "December of the labeled year with January-February of the following year."
        ),
        annual_global_definition=(
            "Annual global means are reconstructed from weighted MAM, JJA, SON, and "
            "DJF means sharing the same labeled year."
        ),
        note=(
            "Net EEI is computed as incoming shortwave minus outgoing shortwave minus "
            "outgoing longwave. all_clear = all-sky minus clear-sky. Radiation trends "
            f"use {RADIATION_TREND_START_YEAR}-{RADIATION_TREND_END_YEAR}, and DJF is "
            "defined by December of the labeled year plus the following January-February."
        ),
    )
    return out


def load_monthly_sst(path: Path = SST_INPUT_FILE) -> xr.DataArray:
    ds = xr.open_dataset(path).sortby("latitude")
    rename_map = {"valid_time": "time", "latitude": "lat", "longitude": "lon"}
    ds = ds.rename(
        {
            key: value
            for key, value in rename_map.items()
            if key in ds.dims or key in ds.coords
        }
    )
    if "number" in ds.dims:
        ds = ds.squeeze("number", drop=True)
    return ds["sst"].sortby("lat")


def build_sst_dataset(input_file: Path = SST_INPUT_FILE) -> xr.Dataset:
    sst = load_monthly_sst(input_file)
    annual_grouped = annual_means(sst)
    annual_trend = fit_trend_map(annual_grouped, time_dim="year")["slope"].astype(
        np.float32
    )
    annual_trend.name = "annual_sst_trend"
    annual_trend.attrs.update(
        long_name="Annual mean sea surface temperature trend",
        units="K yr-1",
    )

    seasonal_trends: list[xr.DataArray] = []
    seasonal_counts: list[int] = []
    seasonal_start_years: list[int] = []
    seasonal_end_years: list[int] = []
    for season in SEASONS:
        grouped = seasonal_means(sst, season)
        seasonal_trends.append(
            fit_trend_map(grouped, time_dim="year")["slope"].expand_dims(
                season=[season]
            )
        )
        seasonal_counts.append(int(grouped.sizes["year"]))
        seasonal_start_years.append(int(grouped["year"].min()))
        seasonal_end_years.append(int(grouped["year"].max()))

    out = xr.Dataset(
        data_vars={
            "annual_sst_trend": annual_trend,
            "seasonal_sst_trend": xr.concat(seasonal_trends, dim="season").astype(
                np.float32
            ),
            "annual_sample_count": xr.DataArray(
                np.int16(annual_grouped.sizes["year"]),
                attrs={"long_name": "Number of annual means used in fit", "units": "1"},
            ),
            "annual_start_year": xr.DataArray(np.int16(annual_grouped["year"].min())),
            "annual_end_year": xr.DataArray(np.int16(annual_grouped["year"].max())),
            "seasonal_sample_count": xr.DataArray(
                np.asarray(seasonal_counts, dtype=np.int16),
                coords={"season": list(SEASONS)},
                dims=("season",),
            ),
            "seasonal_start_year": xr.DataArray(
                np.asarray(seasonal_start_years, dtype=np.int16),
                coords={"season": list(SEASONS)},
                dims=("season",),
            ),
            "seasonal_end_year": xr.DataArray(
                np.asarray(seasonal_end_years, dtype=np.int16),
                coords={"season": list(SEASONS)},
                dims=("season",),
            ),
        },
        coords={
            "lat": annual_trend["lat"],
            "lon": annual_trend["lon"],
            "season": list(SEASONS),
        },
        attrs={
            "title": "Manuscript SST diagnostics",
            "source_file": str(input_file),
            "history": "Created by basic_trend/calculate_manuscript_data.py",
        },
    )
    return out


def build_land_mask_from_sst(
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
    input_file: Path = SST_INPUT_FILE,
) -> xr.DataArray:
    sst = load_monthly_sst(input_file)
    ocean_mask = sst.notnull().any("time").astype(np.float32)

    wrap_lon = float(ocean_mask["lon"].isel(lon=0)) + 360.0
    wrapped_edge = ocean_mask.isel(lon=0).assign_coords(lon=wrap_lon)
    ocean_mask = xr.concat([ocean_mask, wrapped_edge], dim="lon")

    ocean_fraction = ocean_mask.interp(lat=target_lat, lon=target_lon)
    land_mask = ocean_fraction.fillna(0.0) <= 0.5
    land_mask.name = "land_mask"
    land_mask.attrs.update(
        long_name="Land inferred from SST cells missing for all times",
        source_file=str(input_file),
    )
    return land_mask


def regrid_to_target(
    da: xr.DataArray, target_lat: xr.DataArray, target_lon: xr.DataArray
) -> xr.DataArray:
    return da.sortby("lat").interp(
        lat=target_lat,
        lon=target_lon,
        kwargs={"fill_value": "extrapolate"},
    )


def load_mask_drivers() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    omega_ds = xr.open_dataset(OMEGA_FILE)
    omega_seasonal = (
        omega_ds["w"].sel(season=list(SEASONS)).rename(season="period").sortby("lat")
    )
    omega_annual = omega_seasonal.mean("period", skipna=True).expand_dims(
        period=["annual"]
    )
    omega = xr.concat([omega_annual, omega_seasonal], dim="period").assign_coords(
        period=list(PERIODS)
    )
    if STORM_MATRICS == "SLP":
        var = STORM_MATRICS + "_var"
    else:
        var = STORM_MATRICS
    slp_monthly = xr.open_dataset(STORM_FILE)[var].mean("year").sortby("lat")
    slp_fields = [slp_monthly.mean("month", skipna=True).expand_dims(period=["annual"])]
    for season in SEASONS:
        seasonal_mean = slp_monthly.sel(month=SEASON_MONTHS[season]).mean(
            "month", skipna=True
        )
        slp_fields.append(seasonal_mean.expand_dims(period=[season]))
    slp = xr.concat(slp_fields, dim="period").assign_coords(period=list(PERIODS))

    if STORM_MATRICS == "SLP":
        sin_lat = xr.DataArray(
            np.sin(np.deg2rad(slp["lat"].data)),
            coords={"lat": slp["lat"]},
            dims=("lat",),
        )
        sin_lat = xr.where(np.abs(slp["lat"]) < 15.0, 1.0, sin_lat)
        slp = slp / (sin_lat**2)

    ice_ds = xr.open_dataset(ICE_FILE)
    ice_seasonal = (
        ice_ds["siconc"].sel(season=list(SEASONS)).rename(season="period").sortby("lat")
    )
    ice_annual = ice_seasonal.mean("period", skipna=True).expand_dims(period=["annual"])
    ice = xr.concat([ice_annual, ice_seasonal], dim="period").assign_coords(
        period=list(PERIODS)
    )

    return omega, slp, ice


def limits(slp: xr.DataArray, omega: xr.DataArray) -> tuple[float, float, float, float]:
    nh_limit = float(slp.where(slp["lat"] > 0).max(skipna=True)) * STORM_TRACK_NH_FACTOR
    sh_limit = float(slp.where(slp["lat"] < 0).max(skipna=True)) * STORM_TRACK_SH_FACTOR

    omega_pos = omega.where(omega > 0)
    omega_neg = omega.where(omega < 0)
    pos_limit = float(omega_pos.max(skipna=True)) * OMEGA_POS_FACTOR
    neg_limit = float(omega_neg.min(skipna=True)) * OMEGA_NEG_FACTOR
    return nh_limit, sh_limit, pos_limit, neg_limit


def build_region_masks(
    slp: xr.DataArray,
    omega: xr.DataArray,
    ice: xr.DataArray,
    land_mask: xr.DataArray,
) -> dict[str, xr.DataArray]:
    slp_nh_limit, slp_sh_limit, pos_limit, neg_limit = limits(slp, omega)
    lat2d, _ = xr.broadcast(slp["lat"], slp["lon"])
    ice_mask = ice > ICE_FACTOR
    ocean_mask = ~land_mask
    nh_ice = (lat2d > 0) & ice_mask
    sh_ice = ((lat2d < 0) & ice_mask) | (land_mask & (lat2d < -60))

    nh_storm = (lat2d > 0) & (~ice_mask) & (slp >= slp_nh_limit)
    sh_storm = (lat2d < 0) & (~ice_mask) & (slp >= slp_sh_limit)
    storm_track = nh_storm | sh_storm
    equatorward = np.abs(lat2d) < OMEGA_LATITUDE_LIMIT

    nh_positive_omega_ocean = (
        (lat2d > 0)
        & equatorward
        & (~storm_track)
        & (~ice_mask)
        & (omega >= pos_limit)
        & ocean_mask
    )
    sh_positive_omega_ocean = (
        (lat2d < 0)
        & equatorward
        & (~storm_track)
        & (~ice_mask)
        & (omega >= pos_limit)
        & ocean_mask
    )
    tropical_descent = nh_positive_omega_ocean | sh_positive_omega_ocean
    deserts = (
        equatorward & (~storm_track) & (~ice_mask) & (omega >= pos_limit) & land_mask
    )
    tropical_ascent = equatorward & (~storm_track) & (~ice_mask) & (omega <= neg_limit)
    remainder = (
        (~storm_track)
        & (~tropical_descent)
        & (~deserts)
        & (~tropical_ascent)
        & (~nh_ice)
        & (~sh_ice)
    )
    data = [
        nh_ice,
        sh_ice,
        nh_storm,
        sh_storm,
        tropical_descent,
        deserts,
        tropical_ascent,
    ]
    data += [remainder]
    out_dic = {}
    for name, mask in zip(REGION_NAMES, data):
        out_dic[name] = mask
    return out_dic


def weighted_region_statistics(
    data: xr.DataArray,
    mask: xr.DataArray,
    lat_weights: xr.DataArray,
) -> tuple[float, float, float, float]:
    valid_mask = mask & data.notnull()
    weighted_mask = lat_weights.broadcast_like(data).where(valid_mask, 0.0)
    weight_sum = float(weighted_mask.sum(skipna=True))
    if weight_sum == 0.0:
        return np.nan, np.nan, np.nan, 0.0

    weighted_sum = float((data.where(valid_mask, 0.0) * weighted_mask).sum(skipna=True))
    weighted_sq_sum = float(
        ((data.where(valid_mask, 0.0) ** 2) * weighted_mask).sum(skipna=True)
    )

    mean = weighted_sum / weight_sum
    variance = max(weighted_sq_sum / weight_sum - mean**2, 0.0)
    return weighted_sum, mean, float(np.sqrt(variance)), weight_sum


def percent_variability(variability: float, mean: float) -> float:
    if not np.isfinite(variability) or not np.isfinite(mean) or np.isclose(mean, 0.0):
        return np.nan
    return 100.0 * variability / abs(mean)


def build_masks_and_contributions(
    ceres_ds: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset]:
    monthly = load_aligned_ceres_fluxes()
    time_weights = month_length_weights(monthly["all_sky_net"])
    radiation_year_bounds = (RADIATION_TREND_START_YEAR, RADIATION_TREND_END_YEAR)
    seasonal_trend = ceres_ds["all_sky_net_trend"].sel(period=list(SEASONS))
    target_lat = seasonal_trend["lat"]
    target_lon = seasonal_trend["lon"]

    omega, slp, ice = load_mask_drivers()
    seasonal_slp = regrid_to_target(
        slp.sel(period=list(SEASONS)), target_lat, target_lon
    )
    seasonal_omega = regrid_to_target(
        omega.sel(period=list(SEASONS)), target_lat, target_lon
    )
    seasonal_ice = regrid_to_target(
        ice.sel(period=list(SEASONS)), target_lat, target_lon
    )
    land_mask = build_land_mask_from_sst(target_lat, target_lon)

    weights = xr.DataArray(
        np.cos(np.deg2rad(target_lat.data)),
        coords={"lat": target_lat},
        dims=("lat",),
    )

    region_mask_values = np.zeros(
        (len(SEASONS), len(REGION_NAMES), target_lat.size, target_lon.size),
        dtype=np.int8,
    )
    combined_ice = np.zeros(
        (len(SEASONS), target_lat.size, target_lon.size), dtype=np.int8
    )
    combined_storm = np.zeros(
        (len(SEASONS), target_lat.size, target_lon.size), dtype=np.int8
    )
    combined_pos_omega = np.zeros(
        (len(SEASONS), target_lat.size, target_lon.size), dtype=np.int8
    )
    combined_pos_omega_land = np.zeros(
        (len(SEASONS), target_lat.size, target_lon.size), dtype=np.int8
    )
    combined_neg_omega = np.zeros(
        (len(SEASONS), target_lat.size, target_lon.size), dtype=np.int8
    )
    seasonal_fields = {
        season: seasonal_means(
            monthly["all_sky_net"],
            season,
            year_bounds=radiation_year_bounds,
            djf_mode="december_year",
            time_weights=time_weights,
        )
        for season in SEASONS
    }
    seasonal_years = [
        seasonal_fields[season]["year"].values.astype(np.int32) for season in SEASONS
    ]
    common_years = seasonal_years[0]
    for years in seasonal_years[1:]:
        common_years = np.intersect1d(common_years, years)
    common_years = common_years.astype(np.int32)
    season_weights = seasonal_time_fractions(
        monthly["all_sky_net"]["time"], common_years, djf_mode="december_year"
    )

    seasonal_region_series = np.full(
        (len(SEASONS), len(REGION_NAMES), common_years.size), np.nan, dtype=np.float64
    )
    seasonal_region_trend = np.full(
        (len(SEASONS), len(REGION_NAMES)), np.nan, dtype=np.float64
    )
    seasonal_region_ci95 = np.full(
        (len(SEASONS), len(REGION_NAMES)), np.nan, dtype=np.float64
    )
    seasonal_area_fraction = np.full(
        (len(SEASONS), len(REGION_NAMES)), np.nan, dtype=np.float64
    )
    seasonal_contribution_trend = np.full(
        (len(SEASONS), len(REGION_NAMES)), np.nan, dtype=np.float64
    )
    seasonal_contribution_percent = np.full(
        (len(SEASONS), len(REGION_NAMES)), np.nan, dtype=np.float64
    )
    seasonal_sample_count = np.full(
        (len(SEASONS), len(REGION_NAMES)), 0, dtype=np.int16
    )

    annual_region_series = np.full(
        (len(REGION_NAMES), common_years.size), np.nan, dtype=np.float64
    )
    annual_region_trend = np.full(len(REGION_NAMES), np.nan, dtype=np.float64)
    annual_region_ci95 = np.full(len(REGION_NAMES), np.nan, dtype=np.float64)
    annual_contribution_trend = np.full(len(REGION_NAMES), np.nan, dtype=np.float64)
    annual_contribution_percent = np.full(len(REGION_NAMES), np.nan, dtype=np.float64)
    annual_sample_count = np.full(len(REGION_NAMES), 0, dtype=np.int16)
    seasonal_global_series = np.full(
        (len(SEASONS), common_years.size), np.nan, dtype=np.float64
    )
    seasonal_global_trend = np.full(len(SEASONS), np.nan, dtype=np.float64)
    seasonal_global_ci95 = np.full(len(SEASONS), np.nan, dtype=np.float64)
    seasonal_global_contribution_percent = np.full(
        len(SEASONS), np.nan, dtype=np.float64
    )
    seasonal_global_sample_count = np.full(len(SEASONS), 0, dtype=np.int16)
    desert_series = np.full((len(SEASONS), common_years.size), np.nan, dtype=np.float64)
    desert_trend = np.full(len(SEASONS), np.nan, dtype=np.float64)
    desert_ci95 = np.full(len(SEASONS), np.nan, dtype=np.float64)
    desert_area_fraction = np.full(len(SEASONS), np.nan, dtype=np.float64)
    desert_contribution_trend = np.full(len(SEASONS), np.nan, dtype=np.float64)
    desert_contribution_percent = np.full(len(SEASONS), np.nan, dtype=np.float64)
    desert_sample_count = np.full(len(SEASONS), 0, dtype=np.int16)
    annual_desert_series = np.full(common_years.size, np.nan, dtype=np.float64)
    annual_desert_trend = np.nan
    annual_desert_ci95 = np.nan
    annual_desert_contribution_trend = np.nan
    annual_desert_contribution_percent = np.nan
    annual_desert_sample_count = np.int16(0)

    annual_global_series = xr.DataArray(
        np.full(common_years.size, np.nan, dtype=np.float64),
        coords={"year": common_years},
        dims=("year",),
        name="annual_global_series",
    )
    annual_global_trend = np.nan
    annual_global_ci95 = np.nan

    for season_index, season in enumerate(SEASONS):
        season_weight = float(season_weights.sel(season=season))
        masks = build_region_masks(
            seasonal_slp.sel(period=season),
            seasonal_omega.sel(period=season),
            seasonal_ice.sel(period=season),
            land_mask,
        )
        seasonal_field = seasonal_fields[season].sel(year=common_years)
        global_series = area_weighted_global_mean(
            seasonal_field.rename(year="time")
        ).rename(time="year")
        global_series = global_series.assign_coords(year=common_years)
        seasonal_global_series[season_index] = global_series.values
        global_slope, global_ci95, _, global_n_valid = (
            fit_trend_with_confidence_interval(
                global_series,
                coord="year",
            )
        )
        seasonal_global_trend[season_index] = global_slope
        seasonal_global_ci95[season_index] = global_ci95
        seasonal_global_sample_count[season_index] = global_n_valid

        for region_index, region_name in enumerate(REGION_NAMES):
            mask = masks[region_name]
            region_mask_values[season_index, region_index] = mask.astype(np.int8).values
            regional_series = area_weighted_mean_over_mask(seasonal_field, mask)
            seasonal_region_series[season_index, region_index] = regional_series.values
            slope, ci95, _, n_valid = fit_trend_with_confidence_interval(
                regional_series,
                coord="year",
            )
            area_fraction = effective_area_fraction(mask)
            contribution_trend = season_weight * area_fraction * slope

            seasonal_region_trend[season_index, region_index] = slope
            seasonal_region_ci95[season_index, region_index] = ci95
            seasonal_area_fraction[season_index, region_index] = area_fraction
            seasonal_contribution_trend[season_index, region_index] = contribution_trend
            seasonal_sample_count[season_index, region_index] = n_valid
        combined_ice[season_index] = (
            (masks[REGION_NAMES[0]] | masks[REGION_NAMES[1]]).astype(np.int8).values
        )
        combined_storm[season_index] = (
            (masks[REGION_NAMES[2]] | masks[REGION_NAMES[3]]).astype(np.int8).values
        )
        combined_pos_omega[season_index] = (
            masks["Tropical Descent"].astype(np.int8).values
        )
        combined_pos_omega_land[season_index] = masks["Deserts"].astype(np.int8).values
        combined_neg_omega[season_index] = (
            masks["Tropical Ascent"].astype(np.int8).values
        )

        desert_mask = masks["Deserts"]
        desert_regional_series = area_weighted_mean_over_mask(
            seasonal_field, desert_mask
        )
        desert_series[season_index] = desert_regional_series.values
        slope, ci95, _, n_valid = fit_trend_with_confidence_interval(
            desert_regional_series,
            coord="year",
        )
        area_fraction = effective_area_fraction(desert_mask)
        contribution_trend = season_weight * area_fraction * slope

        desert_trend[season_index] = slope
        desert_ci95[season_index] = ci95
        desert_area_fraction[season_index] = area_fraction
        desert_contribution_trend[season_index] = contribution_trend
        desert_sample_count[season_index] = n_valid

    annual_global_series = xr.DataArray(
        np.tensordot(
            season_weights.values.astype(np.float64),
            seasonal_global_series,
            axes=(0, 0),
        ),
        coords={"year": common_years},
        dims=("year",),
        name="annual_global_series",
    )
    if common_years.size >= 3:
        annual_global_trend, annual_global_ci95, _, _ = (
            fit_trend_with_confidence_interval(
                annual_global_series,
                coord="year",
            )
        )

    if np.isfinite(annual_global_trend) and not np.isclose(annual_global_trend, 0.0):
        seasonal_global_contribution_percent[:] = (
            100.0 * season_weights.values * seasonal_global_trend / annual_global_trend
        )
        seasonal_contribution_percent[:] = (
            100.0 * seasonal_contribution_trend / annual_global_trend
        )
        desert_contribution_percent[:] = (
            100.0 * desert_contribution_trend / annual_global_trend
        )

    annual_region_series = np.tensordot(
        season_weights.values.astype(np.float64),
        seasonal_region_series,
        axes=(0, 0),
    )
    for region_index in range(len(REGION_NAMES)):
        annual_series_da = xr.DataArray(
            annual_region_series[region_index],
            coords={"year": common_years},
            dims=("year",),
            name="annual_region_series",
        )
        slope, ci95, _, n_valid = fit_trend_with_confidence_interval(
            annual_series_da,
            coord="year",
        )
        annual_region_trend[region_index] = slope
        annual_region_ci95[region_index] = ci95
        annual_sample_count[region_index] = n_valid
        annual_contribution_trend[region_index] = np.nansum(
            seasonal_contribution_trend[:, region_index]
        )

    annual_desert_series = np.tensordot(
        season_weights.values.astype(np.float64),
        desert_series,
        axes=(0, 0),
    )
    annual_desert_series_da = xr.DataArray(
        annual_desert_series,
        coords={"year": common_years},
        dims=("year",),
        name="annual_desert_series",
    )
    (
        annual_desert_trend,
        annual_desert_ci95,
        _,
        annual_desert_sample_count,
    ) = fit_trend_with_confidence_interval(
        annual_desert_series_da,
        coord="year",
    )
    annual_desert_contribution_trend = np.nansum(desert_contribution_trend)
    if np.isfinite(annual_global_trend) and not np.isclose(annual_global_trend, 0.0):
        annual_contribution_percent[:] = (
            100.0 * annual_contribution_trend / annual_global_trend
        )
        annual_desert_contribution_percent = (
            100.0 * annual_desert_contribution_trend / annual_global_trend
        )

    masks_ds = xr.Dataset(
        data_vars={
            "region_mask": (
                ("season", "region", "lat", "lon"),
                region_mask_values,
            ),
            "ice_mask": (("season", "lat", "lon"), combined_ice),
            "storm_track_mask": (("season", "lat", "lon"), combined_storm),
            "positive_omega_mask": (("season", "lat", "lon"), combined_pos_omega),
            "positive_omega_land_mask": (
                ("season", "lat", "lon"),
                combined_pos_omega_land,
            ),
            "negative_omega_mask": (("season", "lat", "lon"), combined_neg_omega),
        },
        coords={
            "season": list(SEASONS),
            "region": list(REGION_NAMES),
            "lat": target_lat,
            "lon": target_lon,
        },
        attrs={
            "title": "Manuscript circulation regime masks",
            "history": "Created by basic_trend/calculate_manuscript_data.py",
        },
    )

    contributions_ds = xr.Dataset(
        data_vars={
            "seasonal_region_series": (
                ("season", "region", "year"),
                seasonal_region_series,
            ),
            "seasonal_region_trend": (("season", "region"), seasonal_region_trend),
            "seasonal_region_ci95": (("season", "region"), seasonal_region_ci95),
            "seasonal_area_fraction": (("season", "region"), seasonal_area_fraction),
            "seasonal_contribution_trend": (
                ("season", "region"),
                seasonal_contribution_trend,
            ),
            "seasonal_contribution_percent": (
                ("season", "region"),
                seasonal_contribution_percent,
            ),
            "seasonal_sample_count": (("season", "region"), seasonal_sample_count),
            "annual_region_series": (("region", "year"), annual_region_series),
            "annual_region_trend": ("region", annual_region_trend),
            "annual_region_ci95": ("region", annual_region_ci95),
            "annual_contribution_trend": ("region", annual_contribution_trend),
            "annual_contribution_percent": ("region", annual_contribution_percent),
            "annual_sample_count": ("region", annual_sample_count),
            "annual_global_series": (
                "year",
                annual_global_series.values.astype(np.float32),
            ),
            "annual_global_trend": xr.DataArray(np.float32(annual_global_trend)),
            "annual_global_ci95": xr.DataArray(np.float32(annual_global_ci95)),
            "seasonal_global_series": (("season", "year"), seasonal_global_series),
            "seasonal_global_trend": ("season", seasonal_global_trend),
            "seasonal_global_ci95": ("season", seasonal_global_ci95),
            "seasonal_global_contribution_percent": (
                "season",
                seasonal_global_contribution_percent,
            ),
            "seasonal_global_sample_count": ("season", seasonal_global_sample_count),
            "desert_series": (("season", "year"), desert_series),
            "desert_trend": ("season", desert_trend),
            "desert_ci95": ("season", desert_ci95),
            "desert_area_fraction": ("season", desert_area_fraction),
            "desert_contribution_trend": ("season", desert_contribution_trend),
            "desert_contribution_percent": ("season", desert_contribution_percent),
            "desert_sample_count": ("season", desert_sample_count),
            "annual_desert_series": ("year", annual_desert_series),
            "annual_desert_trend": xr.DataArray(np.float32(annual_desert_trend)),
            "annual_desert_ci95": xr.DataArray(np.float32(annual_desert_ci95)),
            "annual_desert_contribution_trend": xr.DataArray(
                np.float32(annual_desert_contribution_trend)
            ),
            "annual_desert_contribution_percent": xr.DataArray(
                np.float32(annual_desert_contribution_percent)
            ),
            "annual_desert_sample_count": xr.DataArray(
                np.int16(annual_desert_sample_count)
            ),
        },
        coords={
            "season": list(SEASONS),
            "region": list(REGION_NAMES),
            "year": common_years,
        },
        attrs={
            "title": "Manuscript regional EEI trend and contribution summary",
            "history": "Created by basic_trend/calculate_manuscript_data.py",
            "radiation_trend_year_range": (
                f"{RADIATION_TREND_START_YEAR}-{RADIATION_TREND_END_YEAR}"
            ),
            "djf_definition": (
                "December of the labeled year with January-February of the following year."
            ),
            "note": (
                "Seasonal contribution percentages are referenced to the annual global-mean "
                "EEI trend reconstructed from the weighted combination of the four "
                "seasonal global series over their common years, so the seasonal "
                "contribution percentages sum to "
                "100%. Contribution trends include both the effective masked area fraction "
                "and the season-specific day weighting derived from the monthly sample. "
                "Annual regional contributions are aggregated from the four seasonal "
                "contributions. Radiation trends use "
                f"{RADIATION_TREND_START_YEAR}-{RADIATION_TREND_END_YEAR}, and DJF is "
                "defined by December of the labeled year plus the following January-February."
            ),
        },
    )

    return masks_ds, contributions_ds


def write_dataset(ds: xr.Dataset, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    encoding = {}
    for name, variable in ds.data_vars.items():
        if np.issubdtype(variable.dtype, np.floating):
            encoding[name] = {"zlib": True, "complevel": 4, "dtype": "float32"}
        elif np.issubdtype(variable.dtype, np.integer):
            encoding[name] = {"zlib": True, "complevel": 4}
    ds.to_netcdf(output_file, engine="netcdf4", encoding=encoding)
    return output_file


def calculate_all(force: bool = True) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ceres_ds = build_ceres_dataset()
    sst_ds = build_sst_dataset()
    masks_ds, contributions_ds = build_masks_and_contributions(ceres_ds)

    paths = {
        "ceres": write_dataset(ceres_ds, CERES_OUTPUT_FILE),
        "masks": write_dataset(masks_ds, MASKS_OUTPUT_FILE),
        "contributions": write_dataset(contributions_ds, CONTRIBUTIONS_OUTPUT_FILE),
        "sst": write_dataset(sst_ds, SST_OUTPUT_FILE),
    }
    return paths


def ensure_manuscript_outputs(force: bool = False) -> dict[str, Path]:
    outputs = {
        "ceres": CERES_OUTPUT_FILE,
        "masks": MASKS_OUTPUT_FILE,
        "contributions": CONTRIBUTIONS_OUTPUT_FILE,
        "sst": SST_OUTPUT_FILE,
    }
    needs_refresh = force or any(not path.exists() for path in outputs.values())
    if not needs_refresh:
        expected_regions = list(REGION_NAMES)
        expected_mask_vars = {
            "region_mask",
            "ice_mask",
            "storm_track_mask",
            "positive_omega_mask",
            "positive_omega_land_mask",
            "negative_omega_mask",
        }
        expected_contribution_vars = {
            "desert_series",
            "desert_trend",
            "desert_ci95",
            "desert_area_fraction",
            "desert_contribution_trend",
            "desert_contribution_percent",
            "desert_sample_count",
            "annual_desert_series",
            "annual_desert_trend",
            "annual_desert_ci95",
            "annual_desert_contribution_trend",
            "annual_desert_contribution_percent",
            "annual_desert_sample_count",
        }
        try:
            contributions_ds = xr.open_dataset(CONTRIBUTIONS_OUTPUT_FILE)
            masks_ds = xr.open_dataset(MASKS_OUTPUT_FILE)
            needs_refresh = (
                list(contributions_ds["region"].values) != expected_regions
                or list(masks_ds["region"].values) != expected_regions
                or not expected_mask_vars.issubset(masks_ds.data_vars)
                or not expected_contribution_vars.issubset(contributions_ds.data_vars)
            )
            contributions_ds.close()
            masks_ds.close()
        except Exception:
            needs_refresh = True
    if needs_refresh:
        return calculate_all(force=True)
    return outputs


def main() -> None:
    paths = calculate_all(force=True)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
