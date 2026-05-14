import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from utils.plotting import central_lon, plot_colormesh, plot_coasts_grid, plot_colorbar, colors, edge_band

masks = xr.open_dataset("pp/regime_masks.nc")
ta = masks.tropical_ascent
mask_union = ta.any(dim="season")
lon_vals = mask_union.lon.values
lat_vals = mask_union.lat.values

ceres = xr.open_mfdataset(["pp/ceres_trends.nc"]).sel(season="ANN")
lwclr_trend = ceres.where(mask_union).lw_clr

drivers_trends = xr.open_dataset('pp/era5_drivers_trends.nc').sel(season="ANN")
tcw_trend = drivers_trends.where(mask_union).tcw
crh_trend = drivers_trends.where(mask_union).column_rh
crh_lwclr_trend = 0 # TODO

weights = np.cos(np.deg2rad(lwclr_trend.lat))
print(xr.corr(lwclr_trend, tcw_trend, dim=("lat","lon"), weights=weights).values)


proj = ccrs.Robinson(central_longitude=central_lon)
fig,axes = plt.subplots(2, 1, figsize=(6,4), subplot_kw={"projection":proj}, constrained_layout=True)

ax = axes[0]
p = plot_colormesh(ax, lwclr_trend)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("a) LW,clr EEI trend / W m$^{-2}$ dec$^{-1}$", loc="left")

ax = axes[1]
p = plot_colormesh(ax, tcw_trend)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
# ax.set_title("b) LW,clr EEI trend / W m$^{-2}$ dec$^{-1}$", loc="left") # TODO
ax.set_title("b) tcw trend / kg m$^{-2}$ dec$^{-1}$", loc="left")

for ax in axes:
    ax.contourf(
        lon_vals, lat_vals, edge_band(mask_union).values,
        levels=[0.5, 1],
        colors=[colors['tropical_ascent']],
        alpha=1,
        transform=ccrs.PlateCarree(),
    )

plot_colorbar(fig, p, "", [0.2, -0.05, 0.6, 0.02])

plt.savefig("figures/clr_OLR.png", dpi=300, facecolor="w", bbox_inches="tight")