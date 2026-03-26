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
    FIGURES_DIR,
    save_figure_outputs,
    CENTRAL_LONGITUDE,
)
from map_plot_utils import wrap_global_field
from trend_utils import fit_trend_map


FIGURE_FILE = FIGURES_DIR / "figure_4_slp_dynamics.png"
SLP_VAR_YEARLY_FILE = SCRIPT_DIR / "output" / "EKE_var_yearly_OnlyTime.nc"
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


def plot_panel(
    ax,
    field: xr.DataArray,
    significant: xr.DataArray,
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
    )
    ax.coastlines(linewidth=0.7)
    ax.set_global()
    ax.set_title(title)
    return contour


def main() -> None:
    ds = xr.open_dataset(SLP_VAR_YEARLY_FILE)
    slp_var = ds["EKE"].sortby("lat") / 1e3

    seasonal_means = {
        season: seasonal_means_from_monthly(slp_var, season) for season in SEASON_MONTHS
    }

    jja = seasonal_means["JJA"]
    mam = seasonal_means["MAM"]
    son = seasonal_means["SON"]
    shoulder_years = np.intersect1d(mam["year"].values, son["year"].values).astype(
        np.int32
    )
    shoulder = mam.sel(year=shoulder_years)
    shoulder = shoulder.assign_coords(year=shoulder_years)

    jja_fit = fit_trend_map(jja, time_dim="year")
    shoulder_fit = fit_trend_map(shoulder, time_dim="year")
    jja_trend = jja_fit["slope"]
    jja_sig = jja_fit["significant"]
    shoulder_trend = shoulder_fit["slope"]
    shoulder_sig = shoulder_fit["significant"]

    jja_trend = jja_trend * 10.0
    shoulder_trend = shoulder_trend * 10.0
    stacked = np.concatenate([jja_trend.values.ravel(), shoulder_trend.values.ravel()])
    finite = np.abs(stacked[np.isfinite(stacked)])
    levels = np.linspace(-50, 50, 21)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=CENTRAL_LONGITUDE)},
    )
    contour = plot_panel(
        axes[0],
        jja_trend,
        jja_sig,
        levels,
        "(a) JJA SLP variance trend",
    )
    plot_panel(
        axes[1],
        shoulder_trend,
        shoulder_sig,
        levels,
        "(b) Shoulder-season SLP variance trend",
    )
    fig.colorbar(
        contour,
        ax=axes,
        orientation="horizontal",
        pad=0.06,
        label=r"hPa$^2$ decade$^{-1}$",
    )
    local_path, overleaf_path = save_figure_outputs(fig, FIGURE_FILE.name, dpi=300)
    plt.close(fig)
    print(local_path)
    print(overleaf_path)


if __name__ == "__main__":
    main()
