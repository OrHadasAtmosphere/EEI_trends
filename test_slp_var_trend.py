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
    seasonal_means as seasonal_means_from_time,
)
from map_plot_utils import wrap_global_field
from trend_utils import fit_trend_map


SLP_VAR_YEARLY_FILE = SCRIPT_DIR / "output" / "SLP_var_yearly.nc"
W_RAW_FILE = SCRIPT_DIR / "output" / "W_raw.nc"
MEAN_FIGURE_FILE = FIGURES_DIR / "test_slp_var_seasonal_means.png"
TREND_FIGURE_FILE = FIGURES_DIR / "test_slp_var_seasonal_trends.png"
SERIES_FIGURE_FILE = FIGURES_DIR / "test_slp_var_hemispheric_series.png"
W_TREND_FIGURE_FILE = FIGURES_DIR / "test_w_seasonal_trends.png"
SEASONS = ("DJF", "MAM", "JJA", "SON")
SEASON_MONTHS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
SEASON_COLORS = {
    "DJF": "#1f77b4",
    "MAM": "#2ca02c",
    "JJA": "#d62728",
    "SON": "#ff7f0e",
}


def seasonal_means_from_monthly(da: xr.DataArray, season: str) -> xr.DataArray:
    members: list[xr.DataArray] = []
    common_years: np.ndarray | None = None

    for month in SEASON_MONTHS[season]:
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


def load_monthly_w(path: Path = W_RAW_FILE) -> xr.DataArray:
    ds = xr.open_dataset(path).sortby("latitude")
    rename_map = {
        "valid_time": "time",
        "latitude": "lat",
        "longitude": "lon",
    }
    ds = ds.rename(
        {
            key: value
            for key, value in rename_map.items()
            if key in ds.dims or key in ds.coords
        }
    )
    if "pressure_level" in ds.dims:
        ds = ds.squeeze("pressure_level", drop=True)
    if "number" in ds.dims:
        ds = ds.squeeze("number", drop=True)
    return ds["w"].sortby("lat")


def area_weighted_mean(field: xr.DataArray, lat_min: float, lat_max: float) -> float:
    region = field.sel(lat=slice(lat_min, lat_max))
    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(region["lat"].values)),
        coords={"lat": region["lat"]},
        dims=("lat",),
    )
    weights_2d = lat_weights.broadcast_like(region)
    numerator = (region * weights_2d).sum(("lat", "lon"), skipna=True)
    denominator = weights_2d.sum(("lat", "lon"), skipna=True)
    return float((numerator / denominator).item())


def area_weighted_mean_series(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
) -> xr.DataArray:
    region = da.sel(lat=slice(lat_min, lat_max))
    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(region["lat"].values)),
        coords={"lat": region["lat"]},
        dims=("lat",),
    )
    weights_3d = lat_weights.broadcast_like(region.isel(year=0, drop=True))
    valid_weights = weights_3d.where(np.isfinite(region))
    numerator = (region * valid_weights).sum(("lat", "lon"), skipna=True)
    denominator = valid_weights.sum(("lat", "lon"), skipna=True)
    return numerator / denominator


def contour_levels(fields: list[xr.DataArray], symmetric: bool) -> np.ndarray:
    values = np.concatenate([field.values.ravel() for field in fields])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(-1.0, 1.0, 21) if symmetric else np.linspace(0.0, 1.0, 21)

    if symmetric:
        vmax = float(np.nanpercentile(np.abs(values), 98))
        if not np.isfinite(vmax) or np.isclose(vmax, 0.0):
            vmax = 1.0
        return np.linspace(-vmax, vmax, 21)

    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmax, vmin):
        vmin, vmax = 0.0, 1.0
    return np.linspace(vmin, vmax, 21)


def plot_panel(ax, field: xr.DataArray, levels: np.ndarray, cmap: str, title: str):
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


def make_figure(
    fields: dict[str, xr.DataArray],
    *,
    filename: str,
    cmap: str,
    levels: np.ndarray,
    unit_label: str,
    title_template: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=60)},
    )

    contour = None
    for index, season in enumerate(SEASONS):
        row, col = divmod(index, 2)
        contour = plot_panel(
            axes[row, col],
            fields[season],
            levels,
            cmap,
            title_template.format(label=f"({chr(97 + index)})", season=season),
        )

    assert contour is not None
    fig.colorbar(
        contour,
        ax=axes,
        orientation="horizontal",
        pad=0.06,
        label=unit_label,
    )
    local_path, overleaf_path = save_figure_outputs(fig, filename, dpi=300)
    plt.close(fig)
    print(local_path)
    print(overleaf_path)


def make_series_figure(
    seasonal_series: dict[str, xr.DataArray],
) -> None:
    hemispheres = (
        ("NH", 0.0, 90.0),
        ("SH", -90.0, 0.0),
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), constrained_layout=True, sharex=True
    )
    for index, (label, lat_min, lat_max) in enumerate(hemispheres):
        ax = axes[index]
        for season in SEASONS:
            series = area_weighted_mean_series(
                seasonal_series[season], lat_min, lat_max
            )
            ax.plot(
                series["year"],
                series,
                color=SEASON_COLORS[season],
                linewidth=2.0,
                label=season,
            )
        ax.set_title(f"({chr(97 + index)}) {label} seasonal mean SLP variance")
        ax.set_ylabel(r"hPa$^2$")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc="best", ncol=4, frameon=False)

    axes[-1].set_xlabel("Year")
    local_path, overleaf_path = save_figure_outputs(
        fig, SERIES_FIGURE_FILE.name, dpi=300
    )
    plt.close(fig)
    print(local_path)
    print(overleaf_path)


def main() -> None:
    ds = xr.open_dataset(SLP_VAR_YEARLY_FILE)
    slp_var = ds["SLP_var"].sortby("lat")
    lat = slp_var["lat"].data[None, None, :, None]
    sin_lat = np.sin(np.deg2rad(lat))
    sin_lat = np.where(np.abs(lat) < 15.0, 1.0, sin_lat)
    slp_var = slp_var / sin_lat / sin_lat
    w = load_monthly_w()

    seasonal_series = {
        season: seasonal_means_from_monthly(slp_var, season) for season in SEASONS
    }
    seasonal_means = {
        season: seasonal_series[season].mean("year", skipna=True) for season in SEASONS
    }
    seasonal_trends = {
        season: fit_trend_map(seasonal_series[season])["slope"] * 10.0
        for season in SEASONS
    }

    print("Area-weighted hemispheric mean SLP variance trends (hPa^2 decade^-1)")
    for season in SEASONS:
        nh = area_weighted_mean(seasonal_trends[season], 0.0, 90.0)
        sh = area_weighted_mean(seasonal_trends[season], -90.0, 0.0)
        print(f"{season}: NH={nh:.3f}, SH={sh:.3f}")

    mean_levels = contour_levels(list(seasonal_means.values()), symmetric=False)
    trend_levels = contour_levels(list(seasonal_trends.values()), symmetric=True)
    seasonal_w_series = {
        season: seasonal_means_from_time(w, season) for season in SEASONS
    }
    seasonal_w_trends = {
        season: fit_trend_map(seasonal_w_series[season])["slope"] * 10.0
        for season in SEASONS
    }
    w_trend_levels = contour_levels(list(seasonal_w_trends.values()), symmetric=True)

    make_figure(
        seasonal_means,
        filename=MEAN_FIGURE_FILE.name,
        cmap="viridis",
        levels=mean_levels,
        unit_label=r"hPa$^2$",
        title_template="{label} {season} mean SLP variance",
    )
    make_figure(
        seasonal_trends,
        filename=TREND_FIGURE_FILE.name,
        cmap="RdBu_r",
        levels=trend_levels,
        unit_label=r"hPa$^2$ decade$^{-1}$",
        title_template="{label} {season} SLP variance trend",
    )
    make_series_figure(seasonal_series)
    make_figure(
        seasonal_w_trends,
        filename=W_TREND_FIGURE_FILE.name,
        cmap="RdBu_r",
        levels=w_trend_levels,
        unit_label=r"Pa s$^{-1}$ decade$^{-1}$",
        title_template="{label} {season} 500 hPa W trend",
    )


if __name__ == "__main__":
    main()
