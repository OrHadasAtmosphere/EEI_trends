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
)
from map_plot_utils import wrap_global_field


FIGURE_FILE = FIGURES_DIR / "figure_2_seasonal_masks.png"
MASK_ALPHA = 0.25


def overlay_masks(ax, masks: xr.Dataset, season: str) -> None:
    for var_name, color in (
        ("ice_mask", "green"),
        ("storm_track_mask", "black"),
        ("positive_omega_mask", "blue"),
        ("negative_omega_mask", "red"),
    ):
        mask = masks[var_name].sel(season=season)
        lon, lat, data = wrap_global_field(mask)
        ax.contourf(
            lon,
            lat,
            data,
            levels=[0.5, 1.5],
            colors=[color],
            alpha=MASK_ALPHA,
            transform=ccrs.PlateCarree(),
        )


def main() -> None:
    ensure_manuscript_outputs(force=False)
    ds = xr.open_dataset(CERES_OUTPUT_FILE)
    masks = xr.open_dataset(MASKS_OUTPUT_FILE)
    start_year = int(ds["start_year"].sel(period="annual"))
    end_year = int(ds["end_year"].sel(period="annual"))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=60)},
    )
    levels = np.linspace(-8, 8, 17)
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
        ax.set_title(f"({panel}) {season}")

    fig.colorbar(contour, ax=axes, orientation="horizontal", pad=0.04, label="W m-2 decade-1",fraction=0.05)
    fig.suptitle(f"Seasonal net EEI trend with circulation masks ({start_year}-{end_year})", y=1.02)
    local_path, overleaf_path = save_figure_outputs(fig, FIGURE_FILE.name, dpi=300)
    plt.close(fig)
    print(local_path)
    print(overleaf_path)


if __name__ == "__main__":
    main()
