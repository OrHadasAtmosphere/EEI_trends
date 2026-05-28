import cartopy.crs as ccrs
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import xarray as xr

from utils.plotting import central_lon, plot_coasts_grid, colors, hatches, edge_band, hatches_legend

ceres = xr.open_mfdataset(["pp/ceres_trends.nc"])
subsidence = xr.open_dataset("pp/regime_masks.nc").subsidence_ocean
drivers_trends = xr.open_dataset('pp/era5_drivers_trends.nc', engine='netcdf4')

def normalize_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0

def lon_idx_selection(data, lon_min, lon_max):
    lon_min_n = normalize_lon(lon_min)
    lon_max_n = normalize_lon(lon_max)
    lon_norm = normalize_lon(data["lon"].values)

    if lon_min_n <= lon_max_n:
        return np.where((lon_norm >= lon_min_n) & (lon_norm <= lon_max_n))[0]
    else:
        return np.where((lon_norm >= lon_min_n) | (lon_norm <= lon_max_n))[0]

def box_selection(data, mask, lat_min, lat_max, lon_min, lon_max):
    """
    selects subsidence ocean mask
    data within a lat-lon box,
    returning numpy array (1D)
    """    
    lon_idx = lon_idx_selection(data, lon_min, lon_max)
    lat_idx = np.where((data["lat"].values >= lat_min) & (data["lat"].values <= lat_max))[0]

    def subselect_and_ravel(da):
        return da.isel(lon=lon_idx, lat=lat_idx).values.ravel()

    data, mask = subselect_and_ravel(data), subselect_and_ravel(mask)
    
    return data[(mask > 0.5)]

def regression(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r, _ = pearsonr(x, y)
    return slope, intercept, r



# define lat-lon boxes containing stcu decks
stcu_deck_boxes = [
    # name, lat_min, lat_max, lon_min, lon_max, color, cmap
    ("Peruvian", -30.0, -10.0, -100.0, -70.0, "purple"),
    ("Namibian", -30.0, -10.0, -15, 15, "red"),
    ("Australian", -35.0, -20.0, 90.0, 115.0, "gray"),
    ("Californian", 10.0, 35.0, -140.0, -110.0,  "green"),
    ("Canarian", 15.0, 30.0, -35.0, -15.0, "blue"),
]

fig = plt.figure(figsize=(9, 12), constrained_layout=True)
ax = fig.add_subplot(2, 1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
map_ax = fig.add_axes(
    [0.0745, 0.745, 0.35, 0.35],  # [left, bottom, width, height]
    projection=ccrs.Robinson(central_longitude=central_lon)
)
map_ax.set_extent([central_lon - 180, central_lon + 180, -60, 60], crs=ccrs.PlateCarree())

# - main plot: - 

ax.set_xlabel("SST Trend / Kdec$^{-1}$")
ax.set_ylabel( "Net CRE Trend / Wm$^{-2}$dec$^{-1}$")

# accumulate all pts for all-deck fit
sst_all_boxes, cre_all_boxes = [], []

for (_, lat_min, lat_max, lon_min, lon_max, color) in stcu_deck_boxes:
    # accumulate per-box for per-box fits
    sst_all_seasons, cre_all_seasons = [], []

    for season in ["MAM", "JJA", "SON", "DJF"]:
        sst_all = drivers_trends.sst.sel(season=season)
        cre_all = ceres.net_cre.sel(season=season)
        region = subsidence.sel(season=season)

        # driver and CRE values within box, subsidence ocean mask
        sst_box = box_selection(sst_all, region, lat_min, lat_max, lon_min, lon_max)
        cre_box = box_selection(cre_all, region, lat_min, lat_max, lon_min, lon_max)

        sst_all_boxes.extend(sst_box)
        cre_all_boxes.extend(cre_box)
        sst_all_seasons.extend(sst_box)
        cre_all_seasons.extend(cre_box)

    # KDE for all points in StCu region, over all seasons    
    sns.kdeplot(x=sst_all_seasons, y=cre_all_seasons, color=color, levels=[0.775, 0.95, 0.99], ax=ax)
    slope, intercept, r = regression(sst_all_seasons, cre_all_seasons)

slope, intercept, r = regression(sst_all_boxes, cre_all_boxes)
x_reg = np.linspace(min(sst_all_boxes), max(sst_all_boxes), 100)
ax.plot(
    x_reg, slope * x_reg + intercept,
    color="black", linestyle="--", linewidth=2,
    label=f"slope$=${slope:.2f} Wm$^{{-2}}$K$^{{-1}}$, $r^2=${r**2:.2f}"
)

xminp, xmaxp, yminp, ymaxp = (5., 92., 6., 90.)
xmin = np.nanpercentile(sst_all_boxes, xminp)
xmax = np.nanpercentile(sst_all_boxes, xmaxp)
ymin = np.nanpercentile(cre_all_boxes, yminp)
ymax = np.nanpercentile(cre_all_boxes, ymaxp)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.legend(frameon=False, loc="lower right", fontsize=12)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(alpha=0.3, linestyle=":")

# - inset plot (map of stcu lat-lon boxes and of subsidence ocean region): -

# color subsidence mask for union of all seasons' masks
mask_union = subsidence.any(dim="season")
lon_vals = mask_union.lon.values
lat_vals = mask_union.lat.values

# 1. Colored boundary band
map_ax.contourf(
    lon_vals, lat_vals, edge_band(mask_union).values,
    levels=[0.5, 1],
    colors=[colors['subsidence_ocean']],
    alpha=0.5,
    transform=ccrs.PlateCarree(),
)

# 2. Hatched regions with colored hatches
map_ax.contourf(
    lon_vals, lat_vals, mask_union.values,
    levels=[0.5, 1],
    colors='none',
    hatches=[hatches['subsidence_ocean']],
    transform=ccrs.PlateCarree(),
)
# trick to color the hatches
map_ax.collections[-1:][0].set_edgecolor(colors['subsidence_ocean'])
map_ax.collections[-1:][0].set_linewidth(0.0)  # remove polygon edges

plot_coasts_grid(map_ax)

# draw filled patches and labels around stcu regions
for name, lat_min, lat_max, lon_min, lon_max, color in stcu_deck_boxes:
    label_lon = lon_min - 0.25 * ((lon_max - lon_min))
    label_lat = lat_max + 0.5
    map_ax.text(
        label_lon, label_lat, name,
        transform=ccrs.PlateCarree(), fontsize=12, ha="left", va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0}, zorder=6,
    )

    data_for_plot = ceres.net_cre.sel(season='ANN')

    if lon_min < 0 and lon_max > 0:
        # split into two if region straddles 0 deg lon
        regions = [(lon_min, 0), (0, lon_max)]
    else:
        regions = [(lon_min, lon_max)]

    for region_lon_min, region_lon_max in regions:
        lon_idx = lon_idx_selection(data_for_plot, region_lon_min, region_lon_max)
        lat_idx = np.where((data_for_plot.lat.values >= lat_min) & (data_for_plot.lat.values <= lat_max))[0]

        masked_array = ma.masked_where(~(mask_union.isel(lon=lon_idx, lat=lat_idx) > 0.5), np.ones_like(mask_union.isel(lon=lon_idx, lat=lat_idx)))
        # use pcolor because contourf gives weird blank line at 0 deg lon
        map_ax.pcolor(
            data_for_plot.lon.values[lon_idx],
            data_for_plot.lat.values[lat_idx],
            masked_array,
            color=color,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )
fig.savefig(f"figures/stcu_ssts.png", dpi=500, bbox_inches='tight')
print(f"plotted sst regressed on net cre for {len(sst_all_boxes)} gridpoints")
