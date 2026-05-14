import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from utils import global_mean
from utils.plotting import central_lon, plot_colormesh, plot_coasts_grid, plot_colorbar

nicenames = {
    "net":"Net,",
    "lw":"LW,",
    "sw":"SW,",
    "_clr":"clr",
    "_cre":"CRE",
    "sst":"Sea surface temperature",
    "tcw":"Total column water vapor",
    "column_rh":"Column relative humidity",
    "siconc":"Sea ice concentration",
}

def eei_and_drivers_maps(eei, driver, savefile):
    proj = ccrs.Robinson(central_longitude=central_lon)
    fig,axes = plt.subplots(4, 3, figsize=(14,9), subplot_kw={"projection":proj}, gridspec_kw={"height_ratios": [1, 1, 0.5, 1],}, constrained_layout=True)
    axes[2,0].remove()
    axes[2,1].remove()
    axes[2,2].remove()
    
    letter = np.array([["a","b","c"],["d","e","f"]])
    for i,vari in enumerate(["_clr","_cre"]):
        for j,varj in enumerate(["net","lw","sw"]):
            ax = axes[i,j]
    
            t = eei[varj+vari]
            trend_gm = global_mean(t).values
            cf = plot_colormesh(ax, t)
    
            plot_coasts_grid(ax)
            Nlabel = f"{nicenames[varj]}{nicenames[vari]}"
            ax.set_title(letter[i,j]+") "+Nlabel+f" = {trend_gm:0.2f}", position=(0.35, 1.0))
    
    plot_colorbar(fig, cf, "EEI trend / W m$^{-2}$ dec$^{-1}$", [0.25, 0.38, 0.5, 0.02])

    letter = np.array(["g","h","i"])
    # vars = ["sst","tcw","siconc"]
    # lim = [1, 2, 0.1]
    # scale = [1, 1, 1]
    # unit = ["K", "kg m$^2$", "%"]
    vars = ["sst","column_rh","siconc"]
    lim = [1, 4, 0.1]
    scale = [1, 100, 1]
    unit = ["K", "%", "%"]
    xbar = [1/6-0.1, 1/2-0.1, 5/6-0.1]
    for i,vari  in enumerate(vars):
        ax = axes[3,i]
        p = plot_colormesh(ax, driver[vari]*scale[i], lim=lim[i], cmap="PRGn")
        plot_coasts_grid(ax)
        ax.set_title(letter[i]+") "+nicenames[vari], position=(0.48, 1.0))
        plot_colorbar(fig, p, "Trend / "+unit[i]+" dec$^{-1}$", [xbar[i], -0.05, 0.2, 0.02], lim=lim[i])
    
    plt.savefig("figures/"+savefile, dpi=300, facecolor="w", bbox_inches="tight")



eei_trend = xr.open_dataset("pp/ceres_trends.nc").sel(season="ANN").load()
drivers_trends = xr.open_dataset("pp/era5_drivers_trends.nc").sel(season="ANN").load()
drivers_trends["siconc"] = drivers_trends["siconc"].where(np.abs(drivers_trends["siconc"]) >= 0.01, np.nan)

eei_and_drivers_maps(eei_trend, drivers_trends, savefile="eei_drivers_maps.png")

