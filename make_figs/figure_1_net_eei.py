import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

from utils.plotting import central_lon, plot_colormesh, plot_coasts_grid, plot_colorbar

ds_trend = xr.open_mfdataset(["pp/ceres_trends.nc"])
ann_trend = ds_trend.sel(season="ANN").load()

ds_ceres_series = xr.open_dataset("pp/ceres_gm_timeseries.nc", engine='netcdf4')
fit = ds_ceres_series.annual_linear_fit
slope, ci = ds_ceres_series.attrs['decadal_net_trend'], ds_ceres_series.attrs['net_trend_ci']

annual_net_gm_toa = ds_ceres_series.sel(season='ANN').net

fig = plt.figure(figsize=(12, 8), constrained_layout=True)
ax_timeseries, ax_map = [
    fig.add_subplot(2, 1, 1),
    fig.add_subplot(2, 1, 2, projection=ccrs.Robinson(central_longitude=central_lon)),
]

ax_timeseries.plot(annual_net_gm_toa.year.values, annual_net_gm_toa.values, color="black", lw=1, label="annual mean")
ax_timeseries.plot(annual_net_gm_toa.year.values, fit, color="red", lw=1,
    label=f"{slope:.2f} ± {ci:.2f} W m$^{-2}$ dec$^{-1}$"
)
ax_timeseries.set_xlabel("Year")
ax_timeseries.set_ylabel("W m$^{-2}$")
ax_timeseries.legend()
ax_timeseries.grid(True, lw=0.5, linestyle=":", color="gray")

colormap = plot_colormesh(ax_map, ann_trend.net)
plot_coasts_grid(ax_map)
plot_colorbar(fig, colormap, "Net EEI trend / W m$^{-2}$ dec$^{-1}$", [0.15, -0.04, 0.7, 0.02])

ax_timeseries.set_title(f"(a) Global mean EEI", loc="center")
ax_map.set_title("(b) Net TOA EEI trend", loc="center")

plt.savefig("figures/figure_1_net_eei.png", dpi=300, facecolor="w", bbox_inches="tight")