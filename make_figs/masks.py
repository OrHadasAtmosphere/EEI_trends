import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import matplotlib.patches as mpatches

from utils import global_mean
from utils.plotting import plot_colormesh, plot_coasts_grid, plot_colorbar
from utils.plotting import colors, hatches, edge_band


net = xr.open_mfdataset(["pp/ceres_trends.nc"]).net.load()
masks = xr.open_dataset("pp/regime_masks.nc").load()

regime_names = list(masks.data_vars)[:-1]
letter = np.array(["a","b","c","d"])

######

fig,axes = plt.subplots(2, 2, figsize=(9.5,6), subplot_kw={"projection":ccrs.Robinson(central_longitude=-135)}, constrained_layout=True)

for i,s in enumerate(["MAM","JJA","SON","DJF"]):
    ax = axes.flatten()[i]
    da = net.sel(season=s)
    trend_gm = global_mean(da).values
    
    cf = plot_colormesh(ax, da)
    plot_coasts_grid(ax)
    ax.set_title(letter[i]+") "+s+f" = {trend_gm:0.2f}", position=(0.35, 1.0))

    for name in regime_names:
        data = masks[name].sel(season=s).astype(int)
        band = edge_band(data, n=2)

        # 1. Colored boundary band (replaces contour)
        ax.contourf(
            data.lon, data.lat, band,
            levels=[0.5, 1],
            colors=[colors[name]],
            transform=ccrs.PlateCarree(),
        )

        # 2. Hatched regions with colored hatches
        ax.contourf(
            data.lon, data.lat, data,
            levels=[0.5, 1],
            colors='none',
            hatches=[hatches[name]],
            transform=ccrs.PlateCarree(),
        )
        # trick to color the hatches
        for collection in ax.collections[-1:]:  # last contourf
            collection.set_edgecolor(colors[name])
            collection.set_linewidth(0.0)  # remove polygon edges
    
plot_colorbar(fig, cf, "Net EEI trend / W m$^{-2}$ dec$^{-1}$", [0.25, -0.05, 0.5, 0.02])
plt.savefig("figures/eei_seasons_masks.png", dpi=300, facecolor="w", bbox_inches="tight")



plt.figure(figsize=(6,4))
bottom = np.zeros(len(masks.season))
regime_names = list(masks.data_vars)[:-1]
for v in regime_names:
    area = (masks[v]*masks.area).sum(["lat","lon"])
    plt.fill_between(masks.season, bottom, bottom+area, color=colors[v], alpha=0.7, label=v)
    plt.plot(masks.season, bottom+area, color="k")
    plt.fill_between(masks.season, bottom, bottom+area, alpha=0, hatch=hatches[v])
    bottom = bottom + area

# Create legend with colored + hatched patches
legend_handles = []
for v in regime_names:
    patch = mpatches.Patch(
        facecolor=colors[v],
        alpha=0.7,
        hatch=hatches[v],
        edgecolor='black',  # optional: outlines the patch
        label=v
    )
    legend_handles.append(patch)
plt.legend(handles=legend_handles, loc=(1,0.5))

plt.ylim(0,5.15e14)
plt.ylabel("Area / m$^2$")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.savefig("figures/area_seasons_masks.png", dpi=300, facecolor="w", bbox_inches="tight")
