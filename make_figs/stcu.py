import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr
import xarray as xr

from utils.plotting import central_lon, plot_coasts_grid, colors, hatches, edge_band

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
    ("Peruvian deck", -30.0, -10.0, -100.0, -70.0, "tab:purple", "Purples"),
    ("Namibian deck", -30.0, -10.0, -15, 15, "tab:red", "Reds"),
    ("Australian deck", -35.0, -20.0, 90.0, 115.0, "tab:gray", "Greys"),
    ("Californian deck", 10.0, 35.0, -140.0, -110.0,  "tab:green", "Greens"),
    ("Canarian deck", 15.0, 30.0, -35.0, -15.0, "tab:blue", "Blues"),
]

fig = plt.figure(figsize=(9, 12), constrained_layout=True)
ax = fig.add_subplot(2, 1, 1)
map_ax = fig.add_subplot(2, 1, 2, projection=ccrs.Robinson(central_longitude=central_lon))

# - upper plot: - 

ax.set_xlabel("SST trend / K decade$^{-1}$")
ax.set_ylabel( "CRE / W m$^{-2}$ decade$^{-1}$")
ax.set_title("SST coupling with Net CRE for StCu decks (all seasons)")

# accumulate all pts for all-deck fit
sst_all_boxes, cre_all_boxes = [], []
legend_patches, contour_labels = [], []

for (name, lat_min, lat_max, lon_min, lon_max, color, cmap_name) in stcu_deck_boxes:
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
    cmap = plt.get_cmap(cmap_name)
    
    sns.kdeplot(x=sst_all_seasons, y=cre_all_seasons, cmap=cmap, levels=[0.5, 0.75, 0.9, 0.95, 0.975, 0.99], ax=ax)
    slope, intercept, r = regression(sst_all_seasons, cre_all_seasons)
    legend_patches.append(mpatches.Patch(color=cmap(0.6), linestyle="-", alpha=0.6))
    contour_labels.append(f"{name}: {slope:.2f} Wm$^{{-2}}$K$^{{-1}}$, R$^2$ = {r**2:.2f}")

slope, intercept, r = regression(sst_all_boxes, cre_all_boxes)
x_reg = np.linspace(min(sst_all_boxes), max(sst_all_boxes), 100)
ax.plot(
    x_reg, slope * x_reg + intercept,
    color=colors['subsidence_ocean'], linestyle="--", linewidth=2,
    label=f"All decks: {slope:.2f} Wm$^{{-2}}$K$^{{-1}}$, R$^2$ = {r**2:.2f}"
)

xminp, xmaxp, yminp, ymaxp = (5., 93., 5., 93.)
xmin = np.nanpercentile(sst_all_boxes, xminp)
xmax = np.nanpercentile(sst_all_boxes, xmaxp)
ymin = np.nanpercentile(cre_all_boxes, yminp)
ymax = np.nanpercentile(cre_all_boxes, ymaxp)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

handles, labels = ax.get_legend_handles_labels()
handles += legend_patches
labels += contour_labels
ax.legend(handles=handles, labels=labels, frameon=False, loc="lower right", fontsize=8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(alpha=0.3, linestyle=":")

# - lower plot (map of stcu lat-lon boxes and of subsidence ocean region): -

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

# draw rectangles and labels around stcu regions
for name, lat_min, lat_max, lon_min, lon_max, color, _ in stcu_deck_boxes:
    lmin = normalize_lon(lon_min)
    lmax = normalize_lon(lon_max)
    rects = [(lmin, lat_min, lmax - lmin, lat_max - lat_min)]
    for x, y, w, h in rects:
        patch = mpatches.Rectangle(
            (x, y), w, h,
            linewidth=1.25, edgecolor=color, facecolor="none",
            transform=ccrs.PlateCarree(), zorder=5,
        )
        map_ax.add_patch(patch)
    label_lon = lmin + 0.5 * ((lmax - lmin))
    label_lat = lat_max
    map_ax.text(
        label_lon, label_lat, name,
        transform=ccrs.PlateCarree(), fontsize=8, ha="left", va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0}, zorder=6,
    )

map_ax.set_global()
map_ax.set_title("Stratocumulus decks within oceanic regions of subsidence (all seasons)")
fig.savefig(f"figures/figure_stcu_ssts.png", dpi=500, bbox_inches='tight')
print(f"plotted sst regressed on net cre for {len(sst_all_boxes)} gridpoints")
