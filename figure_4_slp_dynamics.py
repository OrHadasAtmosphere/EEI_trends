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
    CENTRAL_LONGITUDE,
    FIGURES_DIR,
    SEASONS,
    STORM_MATRICS,
    save_figure_outputs,
)
from map_plot_utils import wrap_global_field
from trend_utils import fit_trend_map

STORM_FILE = "output/surface_winds_monthly.nc"
FIGURE_FILE = FIGURES_DIR / "figure_4_slp_dynamics.png"
SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def seasonal_means_from_monthly(da: xr.DataArray, season: str) -> xr.DataArray:
    months = SEASON_MONTHS[season]
    members: list[xr.DataArray] = []
    common_years: np.ndarray | None = None

    for month in months:
        member = da.sel(month=month).reset_coords("month", drop=True)
        if season == "DJF" and month == 12:
            member = member.assign_coords(year=member["year"] + 1)
        years = member["year"].values.astype(np.int32)
        common_years = (
            years if common_years is None else np.intersect1d(common_years, years)
        )
        members.append(member)

    assert common_years is not None
    aligned = [member.sel(year=common_years) for member in members]
    seasonal = xr.concat(aligned, dim="season_month").mean("season_month", skipna=True)
    return seasonal.assign_coords(year=common_years)


def contour_levels(fields: list[xr.DataArray]) -> np.ndarray:
    values = np.concatenate([field.values.ravel() for field in fields])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(-1.0, 1.0, 21)

    vmax = float(np.nanpercentile(np.abs(values), 98))
    if not np.isfinite(vmax) or np.isclose(vmax, 0.0):
        vmax = 1.0
    return np.linspace(-vmax, vmax, 21)


def colorbar_label(field: xr.DataArray) -> str:
    units = field.attrs.get("units")
    if units:
        return f"{units} decade-1"
    return f"{STORM_MATRICS} decade-1"


def panel_title(index: int, season: str) -> str:
    return f"({chr(97 + index)}) {season} storm-track trend"


def plot_panel(
    ax,
    field: xr.DataArray,
    levels: np.ndarray,
    title: str,
):
    lon, lat, data = wrap_global_field(field)
    contour = ax.contourf(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    ax.coastlines(linewidth=0.7)
    ax.set_global()
    ax.set_title(title)
    return contour


def main() -> None:
    ds = xr.open_dataset(STORM_FILE)
    storm_track = ds["surface_wind"].sortby("lat")
    if STORM_MATRICS == "surface_wind":
        storm_track = storm_track

    seasonal_series = {
        season: seasonal_means_from_monthly(storm_track, season) for season in SEASONS
    }
    seasonal_trends = {
        season: fit_trend_map(seasonal_series[season], time_dim="year")["slope"] * 10.0
        for season in SEASONS
    }
    levels = contour_levels(list(seasonal_trends.values()))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=CENTRAL_LONGITUDE)},
    )

    contour = None
    for index, season in enumerate(SEASONS):
        row, col = divmod(index, 2)
        contour = plot_panel(
            axes[row, col],
            seasonal_trends[season],
            levels,
            panel_title(index, season),
        )

    assert contour is not None
    fig.colorbar(
        contour,
        ax=axes,
        orientation="horizontal",
        pad=0.06,
        label=colorbar_label(storm_track),
    )
    local_path, overleaf_path = save_figure_outputs(fig, FIGURE_FILE.name, dpi=300)
    plt.close(fig)
    print(local_path)
    print(overleaf_path)


if __name__ == "__main__":
    main()
