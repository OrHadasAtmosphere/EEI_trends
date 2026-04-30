import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

from utils.plotting import central_lon, plot_colormesh, plot_coasts_grid, plot_colorbar

drivers_trends = xr.open_dataset('pp/era5_drivers_trends.nc', engine='netcdf4')

fig = plt.figure(figsize=(15, 10), constrained_layout=True)
proj = ccrs.Robinson(central_longitude=central_lon)

ax_a = fig.add_subplot(2, 3, 1, projection=proj)
ax_b = fig.add_subplot(2, 3, 2, projection=proj)
ax_c = fig.add_subplot(2, 3, 3, projection=proj)

sst_lim = 1.
wv_lim = 2.
sic_lim = 0.1

sst_plt = plot_colormesh(ax_a, drivers_trends.sel(season='ANN').sst, lim=sst_lim)
wv_plt = plot_colormesh(ax_b, drivers_trends.sel(season='ANN').tcw, lim=wv_lim)
sic_plt = plot_colormesh(ax_c, drivers_trends.sel(season='ANN').siconc, lim=sic_lim)

plot_coasts_grid(ax_a)
plot_coasts_grid(ax_b)
plot_coasts_grid(ax_c)

ax_a.set_title("(a) SST trend", loc="center")
ax_b.set_title("(b) Total column water vapor trend", loc="center")
ax_c.set_title("(c) Sea ice coverage trend", loc="center")


a_cbar_position = [0.05, 0.575, 0.25, 0.02]  # [left, bottom, width, height]
plot_colorbar(fig, sst_plt, "K dec$^{-1}$", a_cbar_position, lim=sst_lim)

b_cbar_position = [0.38, 0.575, 0.25, 0.02]
plot_colorbar(fig, wv_plt, "kg m$^2$ dec$^{-1}$", b_cbar_position, lim=wv_lim)

c_cbar_position = [0.71, 0.575, 0.25, 0.02]
plot_colorbar(fig, sic_plt, "% dec$^{-1}$", c_cbar_position, lim=sic_lim)

plt.savefig('figures/figure_drivers.png', dpi=300, bbox_inches='tight')