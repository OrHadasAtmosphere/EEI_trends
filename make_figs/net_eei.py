import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

from utils.plotting import central_lon, plot_colormesh, plot_coasts_grid, plot_colorbar

ds_trend = xr.open_mfdataset(["pp/ceres_trends.nc"])
ann_trend = ds_trend.sel(season="ANN").load()

ds_gm = xr.open_dataset("pp/ceres_gm_timeseries.nc").sel(season="ANN")
slope, ci = ds_gm.net_slope_mean, ds_gm.net_slope_ci

fig = plt.figure(figsize=(8, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 1, height_ratios=[2,3])
ax_timeseries, ax_map = [
    fig.add_subplot(gs[0]),
    fig.add_subplot(gs[1], projection=ccrs.Robinson(central_longitude=central_lon)),
]

ax_timeseries.plot(ds_gm.year, ds_gm.net, color="black", marker="o", lw=1)
ax_timeseries.plot(ds_gm.year, ds_gm.net_fit, color="red", lw=1,
    label=f"{ds_gm.net_slope_mean:.2f} $\\pm$ {ds_gm.net_slope_ci:.2f}"+" Wm$^{-2}$dec$^{-1}$"
)
ax_timeseries.set_ylabel("Global-mean Net EEI / W m$^{-2}$")
ax_timeseries.legend(frameon=False)
ax_timeseries.spines["top"].set_visible(False)
ax_timeseries.spines["right"].set_visible(False)

colormap = plot_colormesh(ax_map, ann_trend.net)
plot_coasts_grid(ax_map)
fig.colorbar(colormap, ax=ax_map, orientation="vertical", location="left", 
             shrink=0.8, pad=-0.05, extend="both", ticks = [-5, -2.5, 0, 2.5, 5],
             label="Net EEI trend / W m$^{-2}$ dec$^{-1}$",)

plt.savefig("figures/net_eei.pdf", dpi=300, facecolor="w", bbox_inches="tight")