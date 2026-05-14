import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from utils import global_mean
from utils.plotting import plot_colormesh, plot_coasts_grid, plot_colorbar

def eei_maps(da, savefile):
    fig,axes = plt.subplots(2, 3, figsize=(14,6), subplot_kw={"projection":ccrs.Robinson(central_longitude=-135)}, constrained_layout=True)
    letter = np.array([["a","b","c"],["d","e","f"]])
    for i,vari in enumerate(["_clr","_cre"]):
        for j,varj in enumerate(["net","lw","sw"]):
            ax = axes[i,j]
    
            t = da[varj+vari]
            trend_gm = global_mean(t).values
            cf = plot_colormesh(ax, t)
    
            plot_coasts_grid(ax)
            ax.set_title(letter[i,j]+") "+varj+vari+f" = {trend_gm:0.2f}", position=(0.35, 1.0))
    
    plot_colorbar(fig, cf, "EEI Trend / W m$^{-2}$ dec$^{-1}$", [0.25, -0.05, 0.5, 0.02])
    plt.savefig("figures/"+savefile, dpi=300, facecolor="w", bbox_inches="tight")



ds_trend = xr.open_mfdataset(["pp/ceres_trends.nc"])
da = ds_trend.sel(season="ANN").load()
eei_maps(da, savefile="eei_trend_maps.png")

# for season in ["MAM","JJA","SON","DJF"]:
#     da = ds_trend.sel(season=season).load()
#     eei_maps(da, savefile=f"eei_trend_maps_{season}.png")
