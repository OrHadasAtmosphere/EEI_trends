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
    SST_OUTPUT_FILE,
    save_figure_outputs,
    ensure_manuscript_outputs,
    CENTRAL_LONGITUDE,
)
from map_plot_utils import wrap_global_field
from water_vapour import (
    INPUT_FILE as WV_INPUT_FILE,
    annual_mean_trend as annual_wv_trend,
)


FIGURE_FILE = FIGURES_DIR / "figure_3_drivers.png"


def map_panel(ax, field: xr.DataArray, levels: np.ndarray, cmap: str, title: str):
    lon, lat, data = wrap_global_field(field)
    contour = ax.contourf(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap=cmap,
        extend="both",
    )
    ax.coastlines(linewidth=0.7)
    ax.set_global()
    ax.set_title(title)
    return contour


def blank_panel(ax, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(
        0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=12
    )
    ax.set_axis_off()


def main() -> None:
    ensure_manuscript_outputs(force=False)
    ceres = xr.open_dataset(CERES_OUTPUT_FILE)
    sst = xr.open_dataset(SST_OUTPUT_FILE)

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

    flux_levels = np.linspace(-7.0, 7.0, 15)
    sst_levels = np.linspace(-1.2, 1.2, 13)
    humidity_levels = np.linspace(-2, 2, 9)

    cloud_plot = map_panel(
        axes[0, 0],
        ceres["all_clear_net_trend"].sel(period="annual") * 10.0,
        flux_levels,
        "RdBu_r",
        "(a) Cloud contribution to net EEI trend",
    )
    clear_plot = map_panel(
        axes[0, 1],
        ceres["clear_sky_net_trend"].sel(period="annual") * 10.0,
        flux_levels,
        "RdBu_r",
        "(b) Clear-sky contribution to net EEI trend",
    )
    sst_plot = map_panel(
        axes[1, 0],
        sst["annual_sst_trend"] * 10.0,
        sst_levels,
        "coolwarm",
        "(c) SST trend",
    )
    humidity_plot = None
    if WV_INPUT_FILE.exists():
        humidity_trend, _ = annual_wv_trend(WV_INPUT_FILE)
        humidity_trend = humidity_trend * 10.0
        humidity_plot = map_panel(
            axes[1, 1],
            humidity_trend,
            humidity_levels,
            "BrBG",
            "(d) Total column water vapour trend",
        )
    else:
        blank_panel(
            axes[1, 1],
            "(d) Total column water vapour trend",
            f"Missing input data:\n{WV_INPUT_FILE.name}",
        )

    fig.colorbar(
        cloud_plot,
        ax=[axes[0, 0], axes[0, 1]],
        orientation="horizontal",
        pad=0.05,
        fraction=0.08,
        aspect=50,
        label="W m-2 decade-1",
    )
    fig.colorbar(
        sst_plot,
        ax=axes[1, 0],
        orientation="horizontal",
        pad=0.05,
        label="K decade-1",
        aspect=25,
        fraction=0.08,
    )
    if humidity_plot is not None:
        fig.colorbar(
            humidity_plot,
            ax=axes[1, 1],
            orientation="horizontal",
            pad=0.05,
            label="kg m-2 decade-1",
            aspect=25,
            fraction=0.08,
        )
    local_path, overleaf_path = save_figure_outputs(fig, FIGURE_FILE.name, dpi=300)
    plt.close(fig)
    print(local_path)
    print(overleaf_path)


if __name__ == "__main__":
    main()
