import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from utils import global_mean
from utils.plotting import plot_colormesh, plot_coasts_grid, plot_colorbar

nicenames = {
    "sw_clr":"SW,clr",
    "sw_cre":"SW,CRE",
    "lw_clr":"LW,clr",
    "lw_cre":"LW,CRE",
    "net_clr":"Net,clr",
    "net_cre":"Net,CRE",
}

def eei_maps(da, savefile):
    fig,axes = plt.subplots(6, 4, figsize=(20,18), subplot_kw={"projection":ccrs.Robinson(central_longitude=-135)}, constrained_layout=True)
    letter = np.array([
        ["a", "b", "c", "d"],
        ["e", "f", "g", "h"],
        ["i", "j", "k", "l"],
        ["m", "n", "o", "p"],
        ["q", "r", "s", "t"],
        ["u", "v", "w", "x"]
    ])
    for i,seasoni in enumerate(["MAM","JJA","SON","DJF"]):
        for j,varj in enumerate(["net_clr","sw_clr","lw_clr","net_cre","sw_cre","lw_cre"]):
            ax = axes[j,i]
    
            t = da.sel(season=seasoni)[varj]
            trend_gm = global_mean(t).values
            cf = plot_colormesh(ax, t)
    
            plot_coasts_grid(ax)
            ax.set_title(letter[j,i]+") "+f"{trend_gm:0.2f}", position=(0.32, 1.0))

            if j == 0:
                ax.annotate(seasoni, xy=(0.5, 1.2), xycoords="axes fraction", ha="center", va="center", fontsize=16, fontweight="bold")
            if i == 0:
                ax.annotate(nicenames[varj], xy=(-0.05, 0.5), xycoords="axes fraction", rotation=90, ha="center", va="center", fontsize=16, fontweight="bold")
    
    plot_colorbar(fig, cf, "EEI trend / W m$^{-2}$ dec$^{-1}$", [0.3, -0.02, 0.4, 0.01])
    plt.savefig("figures/"+savefile, dpi=300, facecolor="w", bbox_inches="tight")



ds_trend = xr.open_mfdataset(["pp/ceres_trends.nc"])
da = ds_trend.load()
eei_maps(da, savefile="eei_all_maps.png")
