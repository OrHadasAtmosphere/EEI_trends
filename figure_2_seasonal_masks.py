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
    MASKS_OUTPUT_FILE,
    SEASONS,
    save_figure_outputs,
    ensure_manuscript_outputs,
    CENTRAL_LONGITUDE,
)
from map_plot_utils import wrap_global_field


FIGURE_FILE = FIGURES_DIR / "figure_2_seasonal_masks.png"
MASK_ALPHA = 0.25


def seasonal_trend_levels(
    trend: xr.DataArray,
    *,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    values = np.abs(np.asarray(trend.values, dtype=np.float64))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(-8.0, 8.0, 17)

    limit = float(np.nanpercentile(values, upper_percentile))
    limit = max(1.0, np.ceil(limit))
    return np.linspace(-limit, limit, int(2 * limit) + 1)


def overlay_masks(ax, masks: xr.Dataset, season: str) -> None:
    for var_name, color in (
        ("ice_mask", "green"),
        ("storm_track_mask", "black"),
        ("positive_omega_mask", "blue"),
        ("positive_omega_land_mask", "goldenrod"),
        ("negative_omega_mask", "red"),
    ):
        mask = masks[var_name].sel(season=season)
        lon, lat, data = wrap_global_field(mask)
        ax.contour(
            lon,
            lat,
            data,
            levels=[0.5, 1.5],
            colors=[color],
            alpha=MASK_ALPHA,
            transform=ccrs.PlateCarree(),
            linewidths=5,
        )


def main() -> None:
    ensure_manuscript_outputs(force=False)
    ds = xr.open_dataset(CERES_OUTPUT_FILE)
    masks = xr.open_dataset(MASKS_OUTPUT_FILE)
    period_years = {
        period: (
            int(ds["start_year"].sel(period=period)),
            int(ds["end_year"].sel(period=period)),
        )
        for period in SEASONS
    }
    seasonal_trend = ds["all_sky_net_trend"].sel(period=list(SEASONS)) * 10.0

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        constrained_layout=True,
        subplot_kw={
            "projection": ccrs.Robinson(
                central_longitude=CENTRAL_LONGITUDE,
            )
        },
    )
    levels = seasonal_trend_levels(seasonal_trend)
    contour = None

    for ax, season, panel in zip(axes.flat, SEASONS, ("a", "b", "c", "d"), strict=True):
        trend = ds["all_sky_net_trend"].sel(period=season) * 10.0
        lon, lat, data = wrap_global_field(trend)
        contour = ax.contourf(
            lon,
            lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            levels=levels,
            extend="both",
        )
        overlay_masks(ax, masks, season)
        ax.coastlines(linewidth=0.7)
        ax.set_global()
        start_year, end_year = period_years[season]
        ax.set_title(f"({panel}) {season} ({start_year}-{end_year})")

    fig.colorbar(
        contour,
        ax=axes,
        orientation="horizontal",
        pad=0.04,
        label="W m-2 decade-1",
        fraction=0.05,
    )
    local_path, overleaf_path = save_figure_outputs(fig, FIGURE_FILE.name, dpi=300)
    plt.close(fig)
    print(local_path)
    print(overleaf_path)


if __name__ == "__main__":
    main()
