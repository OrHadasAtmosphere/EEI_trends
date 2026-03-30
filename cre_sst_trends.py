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
from sklearn.neighbors import KernelDensity
from matplotlib.colors import Normalize

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
    ("Peruvian deck", -30.0, -10.0, -100.0, -70.0, "tab:orange"),
    ("Namibian deck", -30.0, -10.0, -15, 15, "tab:green"), 
    #("Australian deck", -35.0, -20.0, 90.0, 115.0, "tab:purple"),
    ("Californian deck", 10.0, 35.0, -140.0, -110.0, "tab:blue"),
    #("Azores", 15.0, 30.0, -35.0, -15.0, "tab:red"),
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

    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 2))
    ax = fig.add_subplot(gs[0, 0])
    map_ax = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=220))

    all_plot_x: list[float] = []
    all_plot_y: list[float] = []

    ax.set_xlabel("SST trend (ERA5) (K decade$^{-1}$)")
    ax.set_ylabel("CRE (CERES) (W m$^{-2}$ decade$^{-1}$)")
    ax.set_title("Regional Stratocumulus-SST coupled radiative feedbacks")

    # prepare density colormap cycle and legend proxies
    DENSITY_CMAPS = ["Oranges", "Greens", "Blues", "Purples", "Reds"]
    density_handles: list = []
    density_labels: list[str] = []

    cloud_annual = ceres[cloud_var].sel(period="annual") * 10.0
    # prefer annual SST trend if available
    if sst_annual_var in sst:
        sst_annual = sst[sst_annual_var] * 10.0
        print("annual SSTs in use")
    else:
        # fall back to mean of seasonal trends
        if sst_seasonal_var in sst:
            # take weighted mean across seasons (equal weights here)
            print("SST trend as annual weighted mean of seasonal")
            sst_annual = sst[sst_seasonal_var].mean(dim="season", skipna=True) * 10.0
        else:
            raise RuntimeError("No SST annual or seasonal trend variable found in SST dataset")

    # build tropical-subsidence mask union and regrid it to the cloud (CERES) grid so we can mask box points
    positive_mask = masks["positive_omega_mask"]
    if "season" in positive_mask.dims:
        mask_union = positive_mask.any(dim="season")
    else:
        mask_union = positive_mask

    for i, (name, lat_min, lat_max, lon_min, lon_max, color) in enumerate(BOX_REGIONS):
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

        mask_sel = mask_union.isel(lon=lon_idx, lat=lat_idx)
        mask_vals = np.asarray(mask_sel.values).ravel()
        
        if sst_sel.size == 0 or cloud_sel.size == 0:
            print(f"{name} lat-lon box returned no gridpoints (sst_sel.size={sst_sel.size}, cloud_sel.size={cloud_sel.size})")
            continue

        sst_vals = np.asarray(sst_sel.values).ravel()
        cloud_vals = np.asarray(cloud_sel.values).ravel()
        # require finite values and that the point lies inside the tropical-subsidence mask
        valid = np.isfinite(sst_vals) & np.isfinite(cloud_vals) & (np.asarray(mask_vals) > 0.5)
        
        xs_pts = sst_vals[valid]
        ys_pts = cloud_vals[valid]

        # standardize to avoid bandwidth issues across axes
        x_mean, x_std = float(np.nanmean(xs_pts)), float(np.nanstd(xs_pts)) or 1.0
        y_mean, y_std = float(np.nanmean(ys_pts)), float(np.nanstd(ys_pts)) or 1.0

        xs_std = (xs_pts - x_mean) / x_std
        ys_std = (ys_pts - y_mean) / y_std
        samples = np.vstack([xs_std, ys_std]).T

        # fit KDE in standardized space; bandwidth chosen empirically
        kde = KernelDensity(bandwidth=0.25, kernel="gaussian")
        kde.fit(samples)

        # build evaluation grid in original units (zoom to cluster extent)
        x_min = float(np.nanpercentile(xs_pts, 1.0))
        x_max = float(np.nanpercentile(xs_pts, 99.0))
        y_min = float(np.nanpercentile(ys_pts, 1.0))
        y_max = float(np.nanpercentile(ys_pts, 99.0))
        # small padding
        pad_x = 0.02 * max(1e-6, x_max - x_min)
        pad_y = 0.02 * max(1e-6, y_max - y_min)
        x_grid = np.linspace(x_min - pad_x, x_max + pad_x, 100)
        y_grid = np.linspace(y_min - pad_y, y_max + pad_y, 100)
        Xg, Yg = np.meshgrid(x_grid, y_grid)
        # convert grid to standardized space for scoring
        grid_std = np.vstack([((Xg.ravel() - x_mean) / x_std), ((Yg.ravel() - y_mean) / y_std)]).T
        log_dens = kde.score_samples(grid_std)
        dens = np.exp(log_dens).reshape(Xg.shape)

        slope, intercept = np.polyfit(xs_pts, ys_pts, 1)
        # plot fit in the high-density region
        dens_thresh = float(np.nanpercentile(dens, 80.0))
        high_mask = (dens >= dens_thresh)
        seg_xmin = float(np.nanmin(Xg[high_mask]))
        seg_xmax = float(np.nanmax(Xg[high_mask]))
        # small padding so line reaches edges of high-density region comfortably
        pad = 0.01 * max(1e-6, seg_xmax - seg_xmin)
        x_line = np.linspace(seg_xmin - pad, seg_xmax + pad, 3)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=2, alpha=0.9, zorder=3)
        print(f"{name}: cluster linear fit slope = {slope:.4f}")

        ax.scatter(xs_pts, ys_pts, s=12, alpha=0.55, color=color, edgecolors="none", zorder=4)

        # normalize density for nicer contour alpha mapping
        norm = Normalize(vmin=np.nanpercentile(dens, 5.0), vmax=np.nanpercentile(dens, 98.0))
        # choose a per-box sequential colormap from the small palette
        cmap_name = DENSITY_CMAPS[i % len(DENSITY_CMAPS)]
        cmap = plt.get_cmap(cmap_name)
        # Mask low-density values
        dens_masked = np.ma.masked_where(dens < np.nanpercentile(dens, 20.0), dens)

        # Make masked values transparent
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_under('none')  # Transparent for values below vmin

        cf = ax.contourf(
            Xg, Yg, dens_masked,
            levels=4,
            cmap=cmap,
            alpha=0.35,
            norm=norm,
            zorder=2,
        )
        # create a proxy patch for the density legend (use a mid-tone from the cmap)
        proxy_color = cmap(0.6)
        density_handles.append(mpatches.Patch(facecolor=proxy_color, edgecolor="none", alpha=0.6))
        density_labels.append(f"{name} feedback: {slope:.2f} Wm$^-2$K$^-1$")

        all_plot_x.extend(xs_pts[np.isfinite(xs_pts)].tolist())
        all_plot_y.extend(ys_pts[np.isfinite(ys_pts)].tolist())

    # set ax limits
    x_arr = np.asarray(all_plot_x, dtype=np.float64)
    y_arr = np.asarray(all_plot_y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    xmin = x_arr.min()
    xmax = float(np.nanpercentile(x_arr[valid], 99.5))
    ymin = y_arr.min()
    ymax = float(np.nanpercentile(y_arr[valid], 99.5))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # add solid black axes at x=0 and y=0
    ax.axvline(0.0, color="k", linewidth=0.3, zorder=2)
    ax.axhline(0.0, color="k", linewidth=0.3, zorder=2)

    # merge existing legend entries (gridpoints / mean / fit lines) with density proxies
    handles, labels = ax.get_legend_handles_labels()
    if density_handles:
        handles = handles + density_handles
        labels = labels + density_labels
    ax.legend(handles=handles, labels=labels, frameon=False, loc="lower right", fontsize=8)
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