import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from utils import global_mean
from utils.plotting import central_lon, plot_colormesh, plot_coasts_grid, plot_colorbar

from matplotlib.colors import LinearSegmentedColormap
colors = ["orange", "white", "cornflowerblue"]
orange_white_blue = LinearSegmentedColormap.from_list("orange_white_blue", colors)

from obs_io import to_trend, add_weights, march_to_feb_years, seasonal_means

nicenames = {
    "net":"Net,",
    "lw":"LW,",
    "sw":"SW,",
    "_clr":"clr",
    "_cre":"CRE",
    "sst":"Sea surface temperature",
    "tcw":"Total column water vapor",
    "column_rh":"Column relative humidity",
    "rh400":"Relative humidity (400 hPa)",
    "crh_600_400":"Relative humidity",
    "siconc":"Sea ice & Aerosol",
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
            ax.set_title(letter[i,j]+") "+Nlabel+f" = {trend_gm:0.2f}", position=(0.22, 1.0), loc="left")
    
    plot_colorbar(fig, cf, "EEI trend / W m$^{-2}$ dec$^{-1}$", [0.25, 0.38, 0.5, 0.02])

    letter = np.array(["g","h","i"])
    # vars = ["sst","tcw","siconc"]
    # lim = [1, 2, 0.1]
    # scale = [1, 1, 1]
    # unit = ["K", "kg m$^2$", "%"]
    vars = ["sst","crh_600_400","siconc"]
    lim = [1, 4, 0.1]
    scale = [1, 100, 1]
    unit = ["K", "%", "%"]
    xbar = [1/6-0.1, 1/2-0.1, 5/6-0.1]

    ax = axes[3,2]
    p = plot_colormesh(ax, driver["aod"], lim=0.1, cmap=orange_white_blue)
    plot_colorbar(fig, p, "AOD Trend / dec$^{-1}$", [5/6+0.01, -0.05, 0.15, 0.02], lim=0.1, nticks=3)
    
    for i,vari  in enumerate(vars):
        ax = axes[3,i]
        p = plot_colormesh(ax, driver[vari]*scale[i], lim=lim[i], cmap="PRGn")
        plot_coasts_grid(ax)
        ax.set_title(letter[i]+") "+nicenames[vari], position=(0.22, 1.0), loc="left")
        if i < 2:
            plot_colorbar(fig, p, "Trend / "+unit[i]+" dec$^{-1}$", [xbar[i], -0.05, 0.2, 0.02], lim=lim[i])
    plot_colorbar(fig, p, "SIC Trend / "+unit[i]+" dec$^{-1}$", [5/6-0.15, -0.05, 0.15, 0.02], lim=lim[i], nticks=3)

    plt.savefig("figures/"+savefile, dpi=300, facecolor="w", bbox_inches="tight")


eei_trend = xr.open_dataset("pp/ceres_trends.nc").sel(season="ANN").load()
drivers_trends = xr.open_dataset("pp/era5_drivers_trends.nc").sel(season="ANN")
drivers_trends["siconc"] = drivers_trends["siconc"].where(np.abs(drivers_trends["siconc"]) >= 0.01, np.nan)

# get aod trend
terra = xr.open_dataset("../CFMIP2024_analysis/MODIS_AOD/Terra_combined.nc").sortby("lat")
terra["lon"] = (terra.coords['lon'] + 360) % 360
terra = terra.sortby("lon").drop_vars("sza")
terra = terra.rename_vars({"aod":"aod_terra"})
newterra = xr.open_dataset("../CFMIP2024_analysis/MODIS_AOD/Terra_2024_2026.nc").sortby("lat").sortby("lon")
newterra = newterra.rename_vars({"aod":"aod_terra"})
terra = terra.combine_first(newterra)

aqua = xr.open_dataset("../CFMIP2024_analysis/MODIS_AOD/Aqua_combined.nc").sortby("lat")
aqua["lon"] = (aqua.coords['lon'] + 360) % 360
aqua = aqua.sortby("lon").drop_vars("sza")
aqua = aqua.rename_vars({"aod":"aod_aqua"})
newaqua = xr.open_dataset("../CFMIP2024_analysis/MODIS_AOD/Aqua_2024_2026.nc").sortby("lat").sortby("lon")
newaqua = newaqua.rename_vars({"aod":"aod_aqua"})
aqua = aqua.combine_first(newaqua)

ds = xr.merge([terra, aqua])
terra.close()
aqua.close()

aod_modis = np.nanmean(np.stack([ds.aod_aqua.values, ds.aod_terra.values]), axis=0)
aod_modis = xr.DataArray(aod_modis, coords=ds.aod_aqua.coords, dims=ds.aod_aqua.dims)
ds = aod_modis.to_dataset(name="aod")
ds = ds.sel(time=slice("2000-03-01", "2026-03-01"))

seas = ds.pipe(add_weights).pipe(march_to_feb_years).pipe(seasonal_means)
trend = to_trend(seas).rename({"aod":"aod_trend"})
seas = seas.rename({"aod":"aod_mean"})
ds = xr.merge([ds.drop_vars("days_in_month"), seas.drop_vars("days_in_month"), trend])

# add to drivers
drivers_trends["aod"] = ds.aod_trend.sel(season="ANN")
drivers_trends = drivers_trends.load()

eei_and_drivers_maps(eei_trend, drivers_trends, savefile="eei_drivers_maps_aod.png")

