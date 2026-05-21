import numpy as np
import xarray as xr
import xesmf as xe
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from obs_io import add_weights, march_to_feb_years, seasonal_means, to_trend
from utils.plotting import central_lon, plot_colormesh, plot_coasts_grid, plot_colorbar, colors, edge_band

masks = xr.open_dataset("pp/regime_masks.nc")
ta = masks.tropical_ascent
mask_union = ta.any(dim="season")
lon_vals = mask_union.lon.values
lat_vals = mask_union.lat.values

lw_clr_ceres = xr.open_dataset("pp/ceres_trends.nc").lw_clr.sel(season="ANN").where(mask_union)
rh_trends = xr.open_dataset("pp/era5_drivers_trends.nc").rh400.sel(season="ANN").where(mask_union)

reconstruct = xr.open_dataset("pp/analytic_lwclr_trends.nc")
lw_clr_varRH = reconstruct.lwclr_varRH.sel(season="ANN").where(mask_union)
lw_clr_varTs = reconstruct.lwclr_varTs.sel(season="ANN").where(mask_union)
lw_clr_varco2 = reconstruct.lwclr_varco2.sel(season="ANN").where(mask_union)
lw_clr_tot = reconstruct.lwclr_all.sel(season="ANN").where(mask_union)

proj = ccrs.Robinson(central_longitude=central_lon)
clim = 3
fig,axes = plt.subplots(6, 1, figsize=(6,12), subplot_kw={"projection":proj}, constrained_layout=True)

ax = axes[0]
p = plot_colormesh(ax, lw_clr_ceres, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("a) CERES observed LW,clr", loc="left")
weights = np.cos(np.deg2rad(lw_clr_ceres.lat))
mean = lw_clr_ceres.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}", loc="right")

ax = axes[1]
p = plot_colormesh(ax, lw_clr_varRH, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("b) RH-component, reconstructed", loc="left")
corr = xr.corr(lw_clr_ceres, lw_clr_varRH, dim=("lat","lon"), weights=weights).values
mean = lw_clr_varRH.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}"+", $r=$"+f"{corr:.2f}", loc="right")

ax = axes[2]
p = plot_colormesh(ax, lw_clr_varTs, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("c) Ts-component, reconstructed", loc="left")
corr = xr.corr(lw_clr_ceres, lw_clr_varTs, dim=("lat","lon"), weights=weights).values
mean = lw_clr_varTs.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}"+", $r=$"+f"{corr:.2f}", loc="right")

ax = axes[3]
p = plot_colormesh(ax, lw_clr_varco2, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("d) CO2-component, reconstructed", loc="left")
corr = xr.corr(lw_clr_ceres, lw_clr_varco2, dim=("lat","lon"), weights=weights).values
mean = lw_clr_varco2.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}"+", $r=$"+f"{corr:.2f}", loc="right")

ax = axes[4]
p = plot_colormesh(ax, lw_clr_tot, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("e) RH & Ts & CO2, reconstructed", loc="left")
corr = xr.corr(lw_clr_ceres, lw_clr_tot, dim=("lat","lon"), weights=weights).values
mean = lw_clr_tot.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}"+", $r=$"+f"{corr:.2f}", loc="right")

ax = axes[5]
resid = lw_clr_ceres - lw_clr_tot
p = plot_colormesh(ax, resid, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("f) Resid", loc="left")
mean = resid.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}", loc="right")

for ax in axes:
    ax.contourf(
        lon_vals, lat_vals, edge_band(mask_union).values,
        levels=[0.5, 1],
        colors=[colors['tropical_ascent']],
        alpha=1,
        transform=ccrs.PlateCarree(),
    )

plot_colorbar(fig, p, "LW,clr EEI trend / W m$^{-2}$ dec$^{-1}$", [0.2, -0.05, 0.6, 0.02], lim=clim)

plt.savefig("figures/clr_OLR_analytic.png", dpi=300, facecolor="w", bbox_inches="tight")
