import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import matplotlib.patches as mpatches

from utils import global_mean
from utils.plotting import plot_coasts_grid
from utils.plotting import colors, hatches, edge_band, hatches_legend, central_lon


net = xr.open_mfdataset(["pp/ceres_trends.nc"]).net.load()
masks = xr.open_dataset("pp/regime_masks.nc").load()

regime_names = list(masks.data_vars)[:-1]
letter = np.array(["a","b","c","d"])

######

fig,axes = plt.subplots(2, 2, figsize=(9.5,6), subplot_kw={"projection":ccrs.Robinson(central_longitude=central_lon)}, constrained_layout=True)

for i,s in enumerate(["MAM","JJA","SON","DJF"]):
    ax = axes.flatten()[i]
    da = net.sel(season=s)
    trend_gm = global_mean(da).values
    
    # cf = plot_colormesh(ax, da)
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
        ax.contourf(
            data.lon, data.lat, data,
            levels=[0.5, 1],
            colors=[colors[name]],
            transform=ccrs.PlateCarree(),
            alpha=0.2,
        )

legend_handles = []
legend_labels = []
for v in regime_names:
    r_formatted = v.replace('_',' ').replace("nh","").replace("sh","")
    r_formatted = r_formatted[0].upper() + r_formatted[1:] if r_formatted[0] != " " else r_formatted[1].upper() + r_formatted[2:]
    patch = mpatches.Patch(
        facecolor=colors[v],
        alpha=0.4,
        edgecolor='black',  # optional: outlines the patch
    )
    # update removing duplicates
    if r_formatted not in legend_labels:
        legend_handles.append(patch)
        legend_labels.append(r_formatted)
fig.legend(
    handles=legend_handles,
    labels=legend_labels,
    ncol=3,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.01),
    bbox_transform=fig.transFigure  # Use figure coordinates
)
    
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
    r_formatted = v.replace('_',' ').replace("nh","NH").replace("sh","SH")
    r_formatted = r_formatted[0].upper() + r_formatted[1:]
    patch = mpatches.Patch(
        facecolor=colors[v],
        alpha=0.7,
        hatch=hatches_legend[v],
        edgecolor='black',  # optional: outlines the patch
        label=r_formatted,
    )
    legend_handles.append(patch)
plt.legend(handles=legend_handles, loc=(1,0.5))

plt.ylim(0,5.15e14)
plt.ylabel("Area / m$^2$")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.savefig("figures/area_seasons_masks.png", dpi=300, facecolor="w", bbox_inches="tight")
