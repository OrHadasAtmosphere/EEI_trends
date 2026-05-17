import numpy as np
import xarray as xr
import xesmf as xe
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from utils.plotting import central_lon, plot_colormesh, plot_coasts_grid, plot_colorbar, colors, edge_band

masks = xr.open_dataset("pp/regime_masks.nc")
ta = masks.tropical_ascent
mask_union = ta.any(dim="season")
lon_vals = mask_union.lon.values
lat_vals = mask_union.lat.values

lw_clr_ceres = xr.open_dataset("pp/ceres_trends.nc").lw_clr.sel(season="ANN").where(mask_union)
lw_clr_varRH = xr.open_dataset("pp/reconstructed_lwclr_trends.nc").lwclr_varRH.sel(season="ANN").where(mask_union)

proj = ccrs.Robinson(central_longitude=central_lon)
fig,axes = plt.subplots(2, 1, figsize=(6,4), subplot_kw={"projection":proj}, constrained_layout=True)

ax = axes[0]
p = plot_colormesh(ax, lw_clr_ceres)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("a) CERES observed LW,clr", loc="left")

ax = axes[1]
p = plot_colormesh(ax, lw_clr_varRH)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("b) RH-component, reconstructed LW,clr", loc="left")

weights = np.cos(np.deg2rad(lw_clr_varRH.lat))
corr = xr.corr(lw_clr_ceres, lw_clr_varRH, dim=("lat","lon"), weights=weights).values
ax.set_title("$r=$"+f"{corr:.2f}", loc="right")

for ax in axes:
    ax.contourf(
        lon_vals, lat_vals, edge_band(mask_union).values,
        levels=[0.5, 1],
        colors=[colors['tropical_ascent']],
        alpha=1,
        transform=ccrs.PlateCarree(),
    )

plot_colorbar(fig, p, "LW,clr EEI trend / W m$^{-2}$ dec$^{-1}$", [0.2, -0.05, 0.6, 0.02])

plt.savefig("figures/clr_OLR.png", dpi=300, facecolor="w", bbox_inches="tight")
