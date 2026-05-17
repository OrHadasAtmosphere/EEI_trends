"""
repo-wide definitions of filenames, plotting norms, and common plotting functions
"""
import cartopy.crs as ccrs
import numpy as np

central_lon = -135
color_range_for_trends = 5

# color per regime
colors = {
    "nh_storms": "grey",
    "nh_cryosphere": "cyan",
    "subsidence_land": "yellow",
    "subsidence_ocean": "lime",
    "tropical_ascent": "purple",
    "sh_storms": "grey",
    "sh_cryosphere": "cyan",
    "residual": "darkorange",
}

# hatching per regime
hatches = {
    "nh_storms": "/",
    "sh_storms": "\\",
    "nh_cryosphere": "/",
    "sh_cryosphere": "\\",
    "subsidence_land": "//",
    "subsidence_ocean": "/",
    "tropical_ascent": "\\",
    "residual": "..",
}

# hatching for legend
hatches_legend = {
    "nh_storms": "//",
    "sh_storms": "\\\\",
    "nh_cryosphere": "//",
    "sh_cryosphere": "\\\\",
    "subsidence_land": "//",
    "subsidence_ocean": "//",
    "tropical_ascent": "\\\\",
    "residual": "..",
}

def plot_colormesh(ax, da, lim=color_range_for_trends, mask_and_val=(None, None), cmap="bwr"):
    mask, val = mask_and_val
    if mask is not None:
        da = da.where(mask == val)
    return ax.pcolormesh(da.lon, da.lat, da, vmin=-lim, vmax=lim,
                        cmap=cmap, transform=ccrs.PlateCarree())

def plot_coasts_grid(ax):
    ax.coastlines(lw=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', linestyle=':')

def plot_colorbar(fig, colormap, title, position, lim=color_range_for_trends, ori="horizontal"):
    cbar_ax = fig.add_axes(position)
    return fig.colorbar(colormap, cax=cbar_ax, orientation=ori, label=title, extend="both", ticks = np.linspace(-lim,lim,5))

def edge_band(mask, n=2):
    """
    n = thickness in grid cells
    """
    interior = mask.astype(bool)

    for _ in range(n):
        north = interior.shift(lat=-1, fill_value=False)
        south = interior.shift(lat=1, fill_value=False)
        east  = interior.roll(lon=-1, roll_coords=False)
        west  = interior.roll(lon=1,  roll_coords=False)

        interior = interior & north & south & east & west

    return mask.astype(bool) & (~interior)
