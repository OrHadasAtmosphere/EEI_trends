import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr
import xarray as xr
import xesmf as xe

from utils.plotting import central_lon, plot_coasts_grid

ceres = xr.open_mfdataset(["pp/ceres_trends.nc"])
lw_clr = ceres.lw_clr
ascent = xr.open_dataset("pp/regime_masks.nc").tropical_ascent
drivers_trends = xr.open_dataset('pp/era5_drivers_trends.nc', engine='netcdf4')
land_sea_mask = (
    xr.open_dataset('raw_data/era5_land_sea_mask.nc')
    .rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'})
    .isel(time=0)
    .lsm
    .drop_vars(['number', 'time', 'expver'])
    .reset_coords(drop=True)
    .drop_attrs()
)

def normalize_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0

def sel_region(data, mask, over_ocean=True):
    """
    selects data within mask, over ocean or land
    returns numpy array of gridpoint values
    """
    data_vals = np.asarray(data.values).ravel()
    mask_vals = np.asarray(mask.values).ravel()

    lon_idx_sel = data.lon.values
    lat_idx_sel = data.lat.values

    land_mask_vals = land_sea_mask.sel(
        lon=lon_idx_sel,
        lat=lat_idx_sel,
        method='nearest'
    ).values.ravel()

    if over_ocean:
        return data_vals[(mask_vals > 0.5) & (land_mask_vals == 0)]
    else:
        return data_vals[(mask_vals > 0.5) & (land_mask_vals == 1)]

def regression(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r, _ = pearsonr(x, y)
    return slope, intercept, r

def plt_kde_with_regression(**kwargs):
    driver, label, title = kwargs.get("data"), kwargs.get("label"), kwargs.get("title")
    over_ocean = kwargs.get("over_ocean")

    driver_regridder = xe.Regridder(driver, ascent, "bilinear", periodic=True)
    lw_regridder = xe.Regridder(lw_clr, ascent, "bilinear", periodic=True)

    fig = plt.figure(figsize=(9, 12), constrained_layout=True)
    ax = fig.add_subplot(2, 1, 1)
    map_ax = fig.add_subplot(2, 1, 2, projection=ccrs.Robinson(central_longitude=central_lon))

    region_type = "ocean" if over_ocean else "land"
    ax.set_title(f"{title} ({region_type} regions)")

    ax.set_xlabel(label)
    ax.set_ylabel("Clear sky OLR (CERES) (W m$^{-2}$ decade$^{-1}$)")

    x_all_seasons, y_all_seasons = [], []

    for season in ["DJF", "MAM", "JJA", "SON"]:
        driver_a = driver_regridder(driver.sel(season=season))
        cloud_a = lw_regridder(lw_clr.sel(season=season))
        region = ascent.sel(season=season)

        x = sel_region(driver_a, region, over_ocean=over_ocean)
        y = sel_region(cloud_a, region, over_ocean=over_ocean)

        x_all_seasons.extend(x)
        y_all_seasons.extend(y)

    # KDE for all points in ascent region, over all seasons
    cmap = plt.get_cmap('Reds') if over_ocean else plt.get_cmap('Greens')
    cmap.set_under('none')
    color = "red" if over_ocean else "green"

    ax.scatter(x_all_seasons, y_all_seasons, s=12, alpha=0.2, color=f"tab:{color}", edgecolors="none")
    sns.kdeplot(x=x_all_seasons, y=y_all_seasons, cmap=cmap, levels=[0.5, 0.75, 0.9, 0.95, 0.975, 0.99], ax=ax)

    # regression for all points (all regions, all seasons)
    slope, intercept, r = regression(x_all_seasons, y_all_seasons)
    x_reg = np.linspace(min(x_all_seasons), max(x_all_seasons), 100)
    ax.plot(
        x_reg, slope * x_reg + intercept,
        color="black", linestyle="--", linewidth=1.5,
        label=f"feedback estimate: {slope:.2f} Wkg$^{{-1}}$, r = {r:.2f}"
    )

    ax_lim_pcts = kwargs.get("ax_lim_percentiles", (5., 93., 5., 93.))
    xminp, xmaxp, yminp, ymaxp = ax_lim_pcts
    xmin = np.nanpercentile(x_all_seasons, xminp)
    xmax = np.nanpercentile(x_all_seasons, xmaxp)
    ymin = np.nanpercentile(y_all_seasons, yminp)
    ymax = np.nanpercentile(y_all_seasons, ymaxp)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, frameon=False, loc=kwargs.get("leg_loc", "lower right"), fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(alpha=0.3, linestyle=":")

    # color mask for union of all seasons' masks
    mask_union = ascent.any(dim="season")
    lon_vals = mask_union.lon.values
    lat_vals = mask_union.lat.values
    mask_data = mask_union.values

    land_mask_map = land_sea_mask.sel(
        lon=lon_vals,
        lat=lat_vals,
        method='nearest'
    ).values

    if over_ocean:
        mask_data = np.where((mask_data > 0.5) & (land_mask_map == 0), 1, 0)
    else:
        mask_data = np.where((mask_data > 0.5) & (land_mask_map == 1), 1, 0)

    map_ax.contourf(
        lon_vals, lat_vals, mask_data,
        levels=[0.5, 1.5],
        colors=["#c1455f"],
        alpha=0.35,
        transform=ccrs.PlateCarree(),
    )
    plot_coasts_grid(map_ax)

    legend_patch = mpatches.Patch(color='#c1455f', alpha=0.35,
                                 label=f'Ascent region ({region_type})')
    ax.legend(handles=[*handles, legend_patch], labels=[*labels, f'Ascent region ({region_type})'],
              frameon=False, loc=kwargs.get("leg_loc", "lower right"), fontsize=8)

    fig.savefig(f"figures/{kwargs.get('savefig')}", dpi=500, bbox_inches='tight')

wv_ocean_params = {
    "data": drivers_trends.tcw,
    "label": "Column water vapor trend (ERA5) (kg m$^{-2}$decade$^{-1}$)",
    "title": "WV coupling with clear sky OLR for ascending regions in tropics (all seasons)",
    "ax_lim_percentiles": (5., 93., 5., 93.),
    "leg_loc": "lower right",
    "over_ocean": True,
    "savefig": "figure_x_wv_olr_ocean.png",
}

wv_land_params = {
    "data": drivers_trends.tcw,
    "label": "Column water vapor trend (ERA5) (kg m$^{-2}$decade$^{-1}$)",
    "title": "WV coupling with clear sky OLR for ascending regions in tropics (all seasons)",
    "ax_lim_percentiles": (5., 93., 5., 93.),
    "leg_loc": "lower right",
    "over_ocean": False,
    "savefig": "figure_x_wv_olr_land.png",
}

plt_kde_with_regression(**wv_ocean_params)
plt_kde_with_regression(**wv_land_params)