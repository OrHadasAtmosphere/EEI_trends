import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def lat_mean(ds):
    return ds.weighted(np.cos(np.deg2rad(ds.lat))).mean("lat")

def eei_maps(da, savefile):
    fig,axes = plt.subplots(2, 3, figsize=(14,6), subplot_kw={"projection":ccrs.Robinson(central_longitude=-135)}, constrained_layout=True)
    letter = np.array([["a","b","c"],["d","e","f"]])
    for i,vari in enumerate(["_clr","_cre"]):
        for j,varj in enumerate(["net","lw","sw"]):
            ax = axes[i,j]
            cr = 5
    
            t = da[varj+vari]
            trend_gm = lat_mean(t.mean("lon")).values
            cf = ax.pcolormesh(da.lon, da.lat, t, vmin=-cr, vmax=cr,
                        cmap="bwr", transform=ccrs.PlateCarree())
    
            ax.coastlines(lw=0.5)
            ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', linestyle=':')
            ax.set_title(letter[i,j]+") "+varj+vari+f" = {trend_gm:0.2f}", position=(0.35, 1.0))
    
    cbar_ax = fig.add_axes([0.25, -0.05, 0.5, 0.02])
    cbar=fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label="EEI Trend / W m$^{-2}$ dec$^{-1}$", extend="both", ticks = np.linspace(-cr,cr,5))
    plt.savefig("figures/"+savefile, dpi=300, facecolor="w", bbox_inches="tight")



ds_trend = xr.open_mfdataset(["pp/ceres_trends.nc"])
da = ds_trend.sel(season="ANN").load()
eei_maps(da, savefile="eei_trend_maps.png")

da = ds_trend.sel(season="MAM").load()
eei_maps(da, savefile="eei_trend_maps_MAM.png")

da = ds_trend.sel(season="JJA").load()
eei_maps(da, savefile="eei_trend_maps_JJA.png")

da = ds_trend.sel(season="SON").load()
eei_maps(da, savefile="eei_trend_maps_SON.png")

da = ds_trend.sel(season="DJF").load()
eei_maps(da, savefile="eei_trend_maps_DJF.png")
