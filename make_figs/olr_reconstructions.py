#packages
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

#data
time_slice = slice('2001-01-01','2025-12-31')
era5_skint = xr.open_dataset('../pp/era5_skint.nc').rename({'valid_time':'time'}).sel(time=time_slice)
era5_rh_400hPa = xr.open_dataset('../pp/era5_rh_400hPa.nc').r.sel(time=time_slice) / 100
ceres_trends = xr.open_dataset('../pp/ceres_trends.nc')
olr_McKim = xr.open_dataset('../pp/McKim_olr.nc')

#functions
def calc_olr(ts_map, rh_map): #interpolate OLR from McKim et.al. 2021
    """
    Calculate outgoing longwave radiation from surface temperature and relative humidity maps.
    
    Parameters
    ----------
    ts_map : xr.DataArray
        Map of surface temperature (K) with any spatial/time dimensions
    rh_map : xr.DataArray
        Map of relative humidity (0-100 or 0-1) with same dimensions as ts_map
        
    Returns
    -------
    olr_map : xr.DataArray
        Map of outgoing longwave radiation (W/m²)
    """
    from scipy.interpolate import interpn
    
    # Ensure inputs are aligned
    ts_map, rh_map = xr.align(ts_map, rh_map, join='override')
    # Get the interpolation grid and values
    # olr is structured as (rh, ts) so we need to transpose
    olr_rh = olr_McKim.olr.rh.values
    olr_ts = olr_McKim.olr.ts.values
    olr_values = olr_McKim.olr.values.T  # Transpose from (rh, ts) to (ts, rh)
    
    # Flatten the input arrays for interpolation
    ts_flat = ts_map.values.ravel()
    rh_flat = rh_map.values.ravel()
    
    # Create points array for interpn - must be in same order as grid
    points = np.column_stack([ts_flat, rh_flat])
    
    # Interpolate using interpn
    olr_flat = interpn(
        (olr_ts, olr_rh),  # Grid points in (ts, rh) order
        olr_values,         # Values transposed to (ts, rh) order
        points,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Reshape back to original shape
    olr_array = olr_flat.reshape(ts_map.shape)
    
    # Restore as DataArray with original coordinates
    olr_map = xr.DataArray(
        olr_array,
        coords=ts_map.coords,
        dims=ts_map.dims,
        name='olr'
    )
    return olr_map

def decadal_trend(da_yearly):
    time_years = da_yearly['time'].dt.year + (da_yearly['time'].dt.dayofyear - 1) / 365.25
    olr_numeric_time = da_yearly.assign_coords(time=time_years)

    trend_per_decade = (
        olr_numeric_time.polyfit(dim='time', deg=1, skipna=True)
        .polyfit_coefficients.sel(degree=1) * 10
    )
    return -1*trend_per_decade

#----------------  calculate Reconstruction of OLR from ERA5 data ---------------- 

#calculate Reconstruction of OLR from ERA5 data

const_ts = era5_skint.skt.mean('time').expand_dims(time=era5_rh_400hPa.time)
const_rh = era5_rh_400hPa.mean('time').expand_dims(time=era5_skint.time)

olr_const_ts_variable_rh = calc_olr(const_ts, era5_rh_400hPa)
olr_variable_ts_const_rh = calc_olr(era5_skint.skt, const_rh)
olr_variable_ts_variable_rh = calc_olr(era5_skint.skt, era5_rh_400hPa)

#calculate pointwise trends from reconstructed OLR

lw_clr_trend_const_ts_variable_rh = decadal_trend(olr_const_ts_variable_rh.resample(time='1YE').mean())
lw_clr_trend_variable_ts_const_rh = decadal_trend(olr_variable_ts_const_rh.resample(time='1YE').mean())
lw_clr_trend_variable_ts_variable_rh = decadal_trend(olr_variable_ts_variable_rh.resample(time='1YE').mean())
lw_clr_trend_ceres = ceres_trends.sel(season='ANN').lw_clr



# ---------------- make supplementary figure ---------------- 

#make supplementary figure

fig, axes = plt.subplots(2, 2, figsize=(14, 5), 
                         subplot_kw={"projection": ccrs.Robinson(central_longitude=0)},
                         constrained_layout=True,dpi=300)
axs = axes.flatten()
vc,hr,co2_hr=0,2,-0.74
cmap = mpl.cm.RdBu_r

# ceres trends

im0 = lw_clr_trend_ceres.plot(
    ax=axs[0],
    transform=ccrs.PlateCarree(),
    cmap='RdBu_r',
    norm = mpl.colors.CenteredNorm(vcenter=vc,halfrange=hr),
    add_colorbar=False
)
axs[0].set_title('CERES LW Clear-Sky Trend')


im1 = lw_clr_trend_const_ts_variable_rh.plot(
    ax=axs[1],
    transform=ccrs.PlateCarree(),
    cmap='RdBu_r',
    norm = mpl.colors.CenteredNorm(vcenter=vc,halfrange=hr),
    add_colorbar=False
)
axs[1].set_title('Reconstruction: Variable RH, Constant Ts')

im2 = (0.74+lw_clr_trend_variable_ts_const_rh).plot(
    ax=axs[2],
    transform=ccrs.PlateCarree(),
    cmap='RdBu_r',
    norm = mpl.colors.CenteredNorm(vcenter=vc,halfrange=hr),
    add_colorbar=False
)
axs[2].set_title('Reconstruction: CO2+Variable Ts, Constant RH')
  
im3 = (0.74+lw_clr_trend_variable_ts_variable_rh).plot(
    ax=axs[3],
    transform=ccrs.PlateCarree(),
    cmap='RdBu_r',
    norm = mpl.colors.CenteredNorm(vcenter=vc,halfrange=hr),
    add_colorbar=False
)
axs[3].set_title('Reconstruction: CO2+Variable Ts, Variable RH')  

for ax in axs:
    ax.coastlines(linewidth=0.5)
    ax.set_extent([-180, 180, -30, 30], ccrs.PlateCarree())

cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
cbar = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal', 
                    label=r'LW Clear-Sky Trend (W m$^{-2}$/decade)', 
                    extend='both')

fig.savefig('../figures/olr_reconstructions.png', dpi=300)
