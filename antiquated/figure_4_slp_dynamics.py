from __future__ import annotations

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from calculate_manuscript_data import (
    CERES_OUTPUT_FILE,
    FIGURES_DIR,
    OUTPUT_DIR,
    SEASON_MONTHS,
    save_figure_outputs,
    ensure_manuscript_outputs,
)
from map_plot_utils import GLOBAL_FONT_SIZE, add_colorbar, wrap_global_field

SEASONS_TO_PLOT = ("MAM", "JJA")
PLOT_EXTENT = (90.0, 350.0, 0.0, 90.0)
FIGURE_SPECS = (
    {
        "storm_file": OUTPUT_DIR / "SLP_var_yearly_OnlyTime.nc",
        "storm_var": "SLP_var",
        "filename": "figure_4_slp_dynamics.png",
        "contour_label": r"SLP variance contours ($\sin^2\phi$ normalized)",
        "normalize_sin2": True,
    },
    {
        "storm_file": OUTPUT_DIR / "EKE_var_yearly_OnlyTime.nc",
        "storm_var": "EKE",
        "filename": "figure_4_slp_dynamics_eke.png",
        "contour_label": "EKE contours",
        "normalize_sin2": False,
    },
)


def seasonal_trend_levels(fields: list[xr.DataArray]) -> np.ndarray:
    values = np.concatenate(
        [np.asarray(field.values, dtype=np.float64).ravel() for field in fields]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(-8.0, 8.0, 17)

    limit = float(np.nanpercentile(np.abs(values), 99.0))
    if not np.isfinite(limit) or np.isclose(limit, 0.0):
        limit = 1.0
    return np.linspace(-limit, limit, 17)


def contour_levels(fields: list[xr.DataArray]) -> np.ndarray:
    values = np.concatenate(
        [np.asarray(field.values, dtype=np.float64).ravel() for field in fields]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, 6)

    lower = float(np.nanpercentile(values, 60.0))
    upper = float(np.nanpercentile(values, 95.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.nanpercentile(values, 40.0))
        upper = float(np.nanpercentile(values, 90.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = 0.0, 1.0
    return np.linspace(lower, upper, 6)


def contour_levels_for_field(
    field: xr.DataArray,
    *,
    n_levels: int,
) -> np.ndarray:
    values = np.asarray(field.values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_levels)

    lower = float(np.nanpercentile(values, 55.0))
    upper = float(np.nanpercentile(values, 97.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.nanpercentile(values, 40.0))
        upper = float(np.nanpercentile(values, 90.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = 0.0, 1.0
    return np.linspace(lower, upper, n_levels)


def apply_sin2_normalization(field: xr.DataArray) -> xr.DataArray:
    sin_lat = xr.DataArray(
        np.sin(np.deg2rad(field["lat"].data)),
        coords={"lat": field["lat"]},
        dims=("lat",),
    )
    sin_lat = xr.where(np.abs(field["lat"]) < 15.0, 1.0, sin_lat)
    return field / (sin_lat**2)


def seasonal_climatology_from_monthly(field: xr.DataArray, season: str) -> xr.DataArray:
    months = SEASON_MONTHS[season]
    return field.sel(month=months).mean(("year", "month"), skipna=True)


def draw_panel(
    ax,
    trend: xr.DataArray,
    trend_levels: np.ndarray,
    storm_mean: xr.DataArray,
    storm_levels: np.ndarray,
    title: str,
):
    lon, lat, data = wrap_global_field(trend)
    filled = ax.contourf(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        levels=trend_levels,
        cmap="RdBu_r",
        extend="both",
    )

    contour_lon, contour_lat, contour_data = wrap_global_field(storm_mean)
    contour = ax.contour(
        contour_lon,
        contour_lat,
        contour_data,
        transform=ccrs.PlateCarree(),
        levels=storm_levels,
        colors="black",
        linewidths=1.0,
    )
    ax.coastlines(linewidth=0.7)
    ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
    ax.set_title(title)
    return filled


def load_eei_trends() -> tuple[dict[str, xr.DataArray], dict[str, tuple[int, int]]]:
    ensure_manuscript_outputs(force=False)
    with xr.open_dataset(CERES_OUTPUT_FILE) as ds:
        trends = {
            season: (ds["all_sky_net_trend"].sel(period=season) * 10.0)
            .sortby("lat")
            .load()
            for season in SEASONS_TO_PLOT
        }
        years = {
            season: (
                int(ds["start_year"].sel(period=season).item()),
                int(ds["end_year"].sel(period=season).item()),
            )
            for season in SEASONS_TO_PLOT
        }
    return trends, years


def load_storm_climatology(
    storm_file: Path,
    storm_var: str,
    *,
    normalize_sin2: bool,
) -> dict[str, xr.DataArray]:
    with xr.open_dataset(storm_file) as ds:
        storm = ds[storm_var].sortby("lat").load()
    if normalize_sin2:
        storm = apply_sin2_normalization(storm)
    return {
        season: seasonal_climatology_from_monthly(storm, season)
        for season in SEASONS_TO_PLOT
    }


def make_overlay_figure(
    eei_trends: dict[str, xr.DataArray],
    period_years: dict[str, tuple[int, int]],
    storm_means: dict[str, xr.DataArray],
    *,
    contour_label: str,
    filename: str,
) -> tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=-135.0)},
    )
    fig.suptitle(contour_label, fontsize=GLOBAL_FONT_SIZE)

    trend_levels = seasonal_trend_levels(
        [eei_trends[season] for season in SEASONS_TO_PLOT]
    )
    storm_levels = {
        season: contour_levels_for_field(
            storm_means[season],
            n_levels=9 if season == "JJA" else 6,
        )
        for season in SEASONS_TO_PLOT
    }

    filled = None
    for index, season in enumerate(SEASONS_TO_PLOT):
        start_year, end_year = period_years[season]
        filled = draw_panel(
            axes[index],
            eei_trends[season],
            trend_levels,
            storm_means[season],
            storm_levels[season],
            f"({chr(97 + index)}) {season} EEI trend ({start_year}-{end_year})",
        )

    assert filled is not None
    add_colorbar(
        fig,
        filled,
        ax=axes,
        orientation="horizontal",
        pad=0.08,
        shrink=0.8,
        aspect=40,
        label=r"EEI trend [W m$^{-2}$ decade$^{-1}$]",
    )
    try:
        local_path, overleaf_path = save_figure_outputs(fig, filename, dpi=300)
    except PermissionError:
        local_path = FIGURES_DIR / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(local_path, dpi=300)
        overleaf_path = local_path
    plt.close(fig)
    return local_path, overleaf_path


def main() -> None:
    eei_trends, period_years = load_eei_trends()
    for spec in FIGURE_SPECS:
        storm_means = load_storm_climatology(
            spec["storm_file"],
            spec["storm_var"],
            normalize_sin2=spec["normalize_sin2"],
        )
        local_path, overleaf_path = make_overlay_figure(
            eei_trends,
            period_years,
            storm_means,
            contour_label=spec["contour_label"],
            filename=spec["filename"],
        )
        print(local_path)
        print(overleaf_path)


if __name__ == "__main__":
    main()
