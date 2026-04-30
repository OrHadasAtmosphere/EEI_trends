import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from utils import global_mean
from utils.plotting import plot_colormesh, plot_coasts_grid, plot_colorbar

def eei_maps(da, savefile):
    fig,axes = plt.subplots(2, 4, figsize=(20,6), subplot_kw={"projection":ccrs.Robinson(central_longitude=-135)}, constrained_layout=True)
    letter = np.array([["a","b","c","d"],["e","f","g","h"]])
    for i,seasoni in enumerate(["MAM","JJA","SON","DJF"]):
        for j,varj in enumerate(["net_clr","net_cre"]):
            ax = axes[j,i]
    
            t = da.sel(season=seasoni)[varj]
            trend_gm = global_mean(t).values
            cf = plot_colormesh(ax, t)
    
            plot_coasts_grid(ax)
            ax.set_title(letter[j,i]+") "+seasoni+", "+varj+f" = {trend_gm:0.2f}", position=(0.35, 1.0))
    
    plot_colorbar(fig, cf, "EEI Trend / W m$^{-2}$ dec$^{-1}$", [0.3, -0.05, 0.4, 0.02])
    plt.savefig("figures/"+savefile, dpi=300, facecolor="w", bbox_inches="tight")



ds_trend = xr.open_mfdataset(["pp/ceres_trends.nc"])
da = ds_trend.load()
eei_maps(da, savefile="eei_seasonal_trend_maps.png")
