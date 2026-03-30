"""
repo-wide definitions of filenames, plotting norms, and common plotting functions
"""
import cartopy.crs as ccrs
import numpy as np

central_lon = -135
color_range_for_trends = 5

def plot_colormesh(ax, da):
    return ax.pcolormesh(da.lon, da.lat, da, vmin=-color_range_for_trends, vmax=color_range_for_trends,
                        cmap="bwr", transform=ccrs.PlateCarree())

def plot_coasts_grid(ax):
    ax.coastlines(lw=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', linestyle=':')

def plot_colorbar(fig, colormap, title, position):
    cbar_ax = fig.add_axes(position)
    return fig.colorbar(colormap, cax=cbar_ax, orientation='horizontal', label=title, extend="both", ticks = np.linspace(-color_range_for_trends,color_range_for_trends,5))

