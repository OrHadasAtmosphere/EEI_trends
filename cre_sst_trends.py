from __future__ import annotations

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

from calculate_manuscript_data import (
    CERES_OUTPUT_FILE,
    SST_OUTPUT_FILE,
    MASKS_OUTPUT_FILE,
    FIGURES_DIR,
    ensure_manuscript_outputs,
    regrid_to_target,
    save_figure_outputs,
)
from map_plot_utils import wrap_global_field

FIGURE_FILE = FIGURES_DIR / "cre_sst_region_scatter.png"

MASK_VARS = [
    ("positive_omega_mask", "blue", "Tropical subsidence"),
]

BOX_REGIONS = [
    # name, lat_min, lat_max, lon_min, lon_max, color
    ("Peruvian deck", -14.0, 0.0, -100.0, -80.0, "tab:brown"),
    ("Chilean deck", -36.5, -15, -100.0, -77.0, "tab:orange"),
    ("Namibian deck", -25.0, -7.0, -12.5, 12.5, "tab:green"), 
    ("Australian deck", -35.0, -15.0, 80.0, 112.0, "tab:purple"),
    ("Californian deck", 14.0, 30.0, -142.0, -110.0, "tab:blue"),
    ("Azores", 20.0, 37.0, -45.0, -15.0, "tab:red"),
]

def main() -> None:
    ensure_manuscript_outputs(force=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ceres = xr.open_dataset(CERES_OUTPUT_FILE)
    sst = xr.open_dataset(SST_OUTPUT_FILE)
    masks = xr.open_dataset(MASKS_OUTPUT_FILE)

    # variable names used in other scripts: all_clear_net_trend (W yr-1, per-year slope), multiply by 10 -> per-decade
    cloud_var = "all_clear_net_trend"
    sst_seasonal_var = "sst_trend"
    sst_annual_var = "annual_sst_trend"

    fig = plt.figure(figsize=(9, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 2))
    ax = fig.add_subplot(gs[0, 0])
    map_ax = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=220))

    all_plot_x: list[float] = []
    all_plot_y: list[float] = []

    ax.set_xlabel("Sea surface temperature trend (K decade$^{-1}$)")
    ax.set_ylabel("Cloud radiative effect (W m$^{-2}$ decade$^{-1}$)")
    ax.set_title("CRE trend vs SST trend")
    
    cloud_annual = ceres[cloud_var].sel(period="annual") * 10.0
    # prefer annual SST trend if available
    if sst_annual_var in sst:
        sst_annual = sst[sst_annual_var] * 10.0
    else:
        # fall back to mean of seasonal trends
        if sst_seasonal_var in sst:
            # take weighted mean across seasons (equal weights here)
            sst_annual = sst[sst_seasonal_var].mean(dim="season", skipna=True) * 10.0
        else:
            raise RuntimeError("No SST annual or seasonal trend variable found in SST dataset")

    for name, lat_min, lat_max, lon_min, lon_max, color in BOX_REGIONS:
        # align sst/cloud to ensure matching coords
        # SST annual trend data
        sst_a, cloud_a = xr.align(sst_annual, cloud_annual, join="inner")

        # Ensure SST is on the same lat/lon grid as the cloud data before aligning.
        # CERES uses half-degree centers (e.g. 0.5, 1.5, ...) while SST here is on integer degrees (0,1,...)
        sst_on_cloud_grid = regrid_to_target(sst_annual, cloud_annual["lat"], cloud_annual["lon"])

        sst_a, cloud_a = xr.align(sst_on_cloud_grid, cloud_annual, join="inner")

        # normalize
        lon_min_n = ((lon_min + 180.0) % 360.0) - 180.0
        lon_max_n = ((lon_max + 180.0) % 360.0) - 180.0

        lon_vals = sst_a["lon"].values
        lon_norm = ((lon_vals + 180.0) % 360.0) - 180.0
        lat_vals = sst_a["lat"].values

        if lon_min_n <= lon_max_n:
            lon_idx = np.where((lon_norm >= lon_min_n) & (lon_norm <= lon_max_n))[0]
        else:
            # box crosses dateline
            lon_idx = np.where((lon_norm >= lon_min_n) | (lon_norm <= lon_max_n))[0]

        lat_idx = np.where((lat_vals >= lat_min) & (lat_vals <= lat_max))[0]

        # select by integer indices (preserves the gridpoint values)
        sst_sel = sst_a.isel(lon=lon_idx, lat=lat_idx)
        cloud_sel = cloud_a.isel(lon=lon_idx, lat=lat_idx)

        if sst_sel.size == 0 or cloud_sel.size == 0:
            print(f"{name} lat-lon box returned no gridpoints (sst_sel.size={sst_sel.size}, cloud_sel.size={cloud_sel.size})")
            continue

        sst_vals = np.asarray(sst_sel.values).ravel()
        cloud_vals = np.asarray(cloud_sel.values).ravel()
        valid = np.isfinite(sst_vals) & np.isfinite(cloud_vals)

        xs_pts = sst_vals[valid]
        ys_pts = cloud_vals[valid]


        slope, intercept = np.polyfit(xs_pts, ys_pts, 1)
        x_min = float(np.nanmin(xs_pts))
        x_max = float(np.nanmax(xs_pts))
        pad = 0.05 * max(1e-6, x_max - x_min)
        x_line = np.linspace(x_min - pad, x_max + pad, 3)
        y_line = slope * x_line + intercept
        ax.scatter(xs_pts, ys_pts, s=12, alpha=0.45, color=color, edgecolors="none", label=f"{name} feedback: {slope:.2f} W m$^{-2}$ K$^{-1}$", zorder=4)
        ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=1.25, alpha=0.9, zorder=3)
        print(f"{name}: cluster linear fit slope = {slope:.4f}")

        all_plot_x.extend(xs_pts[np.isfinite(xs_pts)].tolist())
        all_plot_y.extend(ys_pts[np.isfinite(ys_pts)].tolist())

    x_arr = np.asarray(all_plot_x, dtype=np.float64)
    y_arr = np.asarray(all_plot_y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not valid.any():
        ax.autoscale()
    else:
        # show central 98% of the data (1st to 99th percentile) to zoom in and reduce outlier influence
        xmin = float(np.nanpercentile(x_arr[valid], 1.0))
        xmax = float(np.nanpercentile(x_arr[valid], 99.0))
        ymin = float(np.nanpercentile(y_arr[valid], 1.0))
        ymax = float(np.nanpercentile(y_arr[valid], 99.0))
        # ensure non-zero ranges and add a small padding (2% of the larger range)
        xrng = max(xmax - xmin, 1e-6)
        yrng = max(ymax - ymin, 1e-6)
        pad = 0.02 * max(xrng, yrng)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, linestyle=":")

    # bottom panel: global map showing the tropical-subsidence mask used
    # and the box extents used for the regional box analysis.
    # use union across seasons so the map shows where the regime occurs
    positive_mask = masks["positive_omega_mask"]
    # masks are stored with dim 'season' - take union across seasons for display
    if "season" in positive_mask.dims:
        mask_union = positive_mask.any(dim="season")
    else:
        mask_union = positive_mask

    lon_vals, lat_vals, mask_data = wrap_global_field(mask_union)
    map_ax.contourf(
        lon_vals,
        lat_vals,
        mask_data,
        levels=[0.5, 1.5],
        colors=["#6baed6"],
        alpha=0.35,
        transform=ccrs.PlateCarree(),
    )
    map_ax.coastlines(linewidth=0.6)

    for i, (name, lat_min, lat_max, lon_min, lon_max, color) in enumerate(BOX_REGIONS):
        def norm_lon(lon):
            if lon > 180:
                return lon - 360
            if lon <= -180:
                return ((lon + 180) % 360) - 180
            return lon
        lmin = norm_lon(lon_min)
        lmax = norm_lon(lon_max)
        # handle boxes that cross the dateline by splitting drawing into up to two rects
        if lmin <= lmax:
            rects = [ (lmin, lat_min, lmax - lmin, lat_max - lat_min) ]
        else:
            rects = [
                (lmin, lat_min, 180.0 - lmin, lat_max - lat_min),
                (-180.0, lat_min, lmax + 180.0, lat_max - lat_min),
            ]
        for x, y, w, h in rects:
            patch = mpatches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1.25,
                edgecolor=color,
                facecolor="none",
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            map_ax.add_patch(patch)
        label_lon = lmin + 0.5 * ( (lmax - lmin) if lmin <= lmax else ( (180.0 - lmin) - ( -180.0 - lmax ) ) )
        label_lat = lat_max
        map_ax.text(
            label_lon,
            label_lat,
            name,
            transform=ccrs.PlateCarree(),
            fontsize=9,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
            zorder=6,
        )

    map_ax.set_global()
    map_ax.set_title("Stratocumulus decks within regions of subsidence")

    save_figure_outputs(fig, FIGURE_FILE.name, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()