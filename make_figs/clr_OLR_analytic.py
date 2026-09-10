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
weights = np.cos(np.deg2rad(lw_clr_ceres.lat))

rh_var = "crh_600_400"

reconstruct = xr.open_dataset("pp/analytic_lwclr_trends_"+rh_var+".nc")
lw_clr_varRH = reconstruct.lwclr_varRH.sel(season="ANN").where(mask_union)
lw_clr_varTs = reconstruct.lwclr_varTs.sel(season="ANN").where(mask_union)
lw_clr_varco2 = reconstruct.lwclr_varco2.sel(season="ANN").where(mask_union)
lw_clr_tot = reconstruct.lwclr_all.sel(season="ANN").where(mask_union)
resid = lw_clr_ceres - lw_clr_tot

print("correlation between RH-component and residual")
print(xr.corr(lw_clr_varRH, resid, dim=("lat","lon"), weights=weights).values)

proj = ccrs.Robinson(central_longitude=central_lon)
clim = 3
fig,axes = plt.subplots(3, 2, figsize=(12,6), subplot_kw={"projection":proj}, constrained_layout=True)

ax = axes[0,0]
p = plot_colormesh(ax, lw_clr_ceres, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("a) CERES observed LW,clr", loc="left")
mean = lw_clr_ceres.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}", loc="right")

ax = axes[1,0]
p = plot_colormesh(ax, lw_clr_varRH, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("c) RH-component, analytic", loc="left")
corr = xr.corr(lw_clr_ceres, lw_clr_varRH, dim=("lat","lon"), weights=weights).values
mean = lw_clr_varRH.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}"+", $r=$"+f"{corr:.2f}", loc="right")

ax = axes[2,0]
p = plot_colormesh(ax, lw_clr_varTs, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("e) $T_s$-component, analytic", loc="left")
corr = xr.corr(lw_clr_ceres, lw_clr_varTs, dim=("lat","lon"), weights=weights).values
mean = lw_clr_varTs.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}"+", $r=$"+f"{corr:.2f}", loc="right")

ax = axes[0,1]
p = plot_colormesh(ax, lw_clr_tot, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("b) RH & $T_s$ & CO$_2$, analytic", loc="left")
corr = xr.corr(lw_clr_ceres, lw_clr_tot, dim=("lat","lon"), weights=weights).values
mean = lw_clr_tot.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}"+", $r=$"+f"{corr:.2f}", loc="right")

ax = axes[1,1]
p = plot_colormesh(ax, lw_clr_varco2, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("d) CO$_2$-component, analytic", loc="left")
corr = xr.corr(lw_clr_ceres, lw_clr_varco2, dim=("lat","lon"), weights=weights).values
mean = lw_clr_varco2.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}"+", $r=$"+f"{corr:.2f}", loc="right")

ax = axes[2,1]
p = plot_colormesh(ax, resid, lim=clim)
plot_coasts_grid(ax)
ax.set_extent([-180, 180, -45, 45],crs=ccrs.PlateCarree())
ax.set_title("f) Residual", loc="left")
corr = xr.corr(lw_clr_ceres, resid, dim=("lat","lon"), weights=weights).values
mean = resid.mean("lon").weighted(weights).mean("lat").values
ax.set_title("mean$=$"+f"{mean:.2f}"+", $r=$"+f"{corr:.2f}", loc="right")

for ax in axes.flatten():
    ax.contourf(
        lon_vals, lat_vals, edge_band(mask_union).values,
        levels=[0.5, 1],
        colors=[colors['tropical_ascent']],
        alpha=1,
        transform=ccrs.PlateCarree(),
    )

plot_colorbar(fig, p, "LW,clr EEI trend / W m$^{-2}$ dec$^{-1}$", [0.3, -0.05, 0.4, 0.02], lim=clim)

plt.savefig("figures/clr_OLR_analytic_"+rh_var+".pdf", dpi=300, facecolor="w", bbox_inches="tight")
