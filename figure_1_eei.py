from __future__ import annotations

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

from calculate_manuscript_data import (
    CERES_OUTPUT_FILE,
    FIGURES_DIR,
    CENTRAL_LONGITUDE,
    save_figure_outputs,
    ensure_manuscript_outputs,
)
from map_plot_utils import wrap_global_field


FIGURE_FILE = FIGURES_DIR / "figure_1_eei.png"
LATITUDE_LINES = np.arange(-75, 76, 15)


def format_latitude_label(latitude: float) -> str:
    if latitude > 0:
        return f"{int(abs(latitude))}\N{DEGREE SIGN}N"
    if latitude < 0:
        return f"{int(abs(latitude))}\N{DEGREE SIGN}S"
    return "0\N{DEGREE SIGN}"


def add_latitude_lines(ax, c_lon) -> None:
    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(central_longitude=c_lon),
        draw_labels=False,
        linewidth=0.6,
        color="0.35",
        alpha=0.7,
        linestyle=":",
    )
    gridlines.xlocator = mticker.FixedLocator([])
    gridlines.ylocator = mticker.FixedLocator(LATITUDE_LINES)

    for latitude in LATITUDE_LINES:
        ax.text(
            -314.99,
            float(latitude),
            format_latitude_label(float(latitude)),
            transform=ccrs.PlateCarree(central_longitude=c_lon),
            ha="right",
            va="center",
            fontsize=6,
            color="0.25",
            clip_on=False,
        )


def draw_map(ax, field: xr.DataArray, levels: np.ndarray, c_lon: int, cmap: str, title: str):
    lon, lat, data = wrap_global_field(field)
    contour = ax.contourf(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap=cmap,
    )
    ax.coastlines(linewidth=0.7)
    add_latitude_lines(ax, c_lon)
    ax.set_global()
    ax.set_title(title)
    return contour


def main() -> None:
    ensure_manuscript_outputs(force=False)
    ds = xr.open_dataset(CERES_OUTPUT_FILE)

    eei_series = ds["all_sky_global_net_annual_mean"]
    eei_fit = ds["all_sky_global_net_annual_fit"]
    eei_trend = ds["all_sky_net_trend"].sel(period="annual") * 10.0
    start_year = int(eei_series["year"].min())
    end_year = int(eei_series["year"].max())
    trend_decade = float(ds["all_sky_global_net_annual_trend"]) * 10.0
    # try to read 95% CI (per year) and convert to per-decade for annotation
    trend_ci95_decade = None
    raw_ci = ds["all_sky_global_net_annual_ci95"].values
    if np.isfinite(raw_ci):
        trend_ci95_decade = float(raw_ci) * 10.0

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 6), constrained_layout=True)
    c_lon = 220
    axes = [
        fig.add_subplot(2, 1, 1),
        fig.add_subplot(2, 1, 2, projection=ccrs.Robinson(central_longitude=CENTRAL_LONGITUDE)),
    ]

    trend_limit = 5
    if not np.isfinite(trend_limit) or np.isclose(trend_limit, 0.0):
        trend_limit = 1.0

    axes[0].plot(
        eei_series["year"],
        eei_series,
        color="0.2",
        linewidth=2,
        label="Annual mean",
    )
    axes[0].plot(
        eei_fit["year"],
        eei_fit,
        color="tab:red",
        linewidth=2,
        label=rf"Trend: {trend_decade:.2f} ± {trend_ci95_decade:.1f} W m$^{{-2}}$ decade$^{{-1}}$",
    )
    axes[0].set_title(f"(a) Global-mean EEI ({start_year}-{end_year})")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel(r"W m$^{-2}$")
    axes[0].grid(alpha=0.3, linestyle=":")
    axes[0].legend(frameon=False, loc="upper left")
    trend_plot = draw_map(
        axes[1],
        eei_trend,
        np.linspace(-trend_limit, trend_limit, 11),
        c_lon,
        "RdBu_r",
        f"(b) EEI trend ({start_year}-{end_year})",
    )

    fig.colorbar(
        trend_plot,
        ax=axes[1],
        orientation="horizontal",
        pad=0.05,
        shrink=0.6,
        aspect=40,
        label=r"W m$^{-2}$ decade$^{-1}$",
    )
    local_path, overleaf_path = save_figure_outputs(fig, FIGURE_FILE.name, dpi=300)
    plt.close(fig)
    print(local_path)
    print(overleaf_path)


if __name__ == "__main__":
    main()
