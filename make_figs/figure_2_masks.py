import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from utils import global_mean
from utils.plotting import plot_colormesh, plot_coasts_grid, plot_colorbar

net = xr.open_mfdataset(["pp/ceres_trends.nc"]).net.load()

fig,axes = plt.subplots(2, 2, figsize=(9.5,6), subplot_kw={"projection":ccrs.Robinson(central_longitude=-135)}, constrained_layout=True)
letter = np.array(["a","b","c","d"])

for i,s in enumerate(["MAM","JJA","SON","DJF"]):
    ax = axes.flatten()[i]
    da = net.sel(season=s)
    trend_gm = global_mean(da).values
    
    cf = plot_colormesh(ax, da)
    plot_coasts_grid(ax)
    ax.set_title(letter[i]+") "+s+f" = {trend_gm:0.2f}", position=(0.35, 1.0))
    
plot_colorbar(fig, cf, "EEI Trend / W m$^{-2}$ dec$^{-1}$", [0.25, -0.05, 0.5, 0.02])
plt.savefig("figures/eei_seasons_masks.png", dpi=300, facecolor="w", bbox_inches="tight")
