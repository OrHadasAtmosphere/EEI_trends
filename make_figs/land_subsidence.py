import numpy as np
import xarray as xr
import xesmf as xe
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


from obs_io import add_weights, march_to_feb_years, seasonal_means, to_trend
from utils.plotting import central_lon, plot_colormesh, plot_coasts_grid, plot_colorbar, colors, edge_band
from utils import trend_and_ci

masks = xr.open_dataset("pp/regime_masks.nc")
lsm = xr.open_dataset("pp/era5_clim.nc").lsm.isel(season=0)

ceres = xr.open_dataset('raw_data/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202602.nc')
ceres = ceres.sel(time=slice("2000-03-01", "2026-03-01"))
ceres = add_weights(ceres)
ceres = march_to_feb_years(ceres)
ceres= seasonal_means(ceres)
#Calculate trends in the land subsidence 

st = masks.subsidence_land
mask_union = st.any(dim="season")
lon_vals = mask_union.lon.values
lat_vals = mask_union.lat.values
weights = masks.area.sel(season='MAM') #same for all seasons

sw_clr_ceres = xr.open_dataset("../pp/ceres_trends.nc").sw_clr.sel(season="ANN").where(mask_union)
net_clr_ceres = xr.open_dataset("../pp/ceres_trends.nc").net_clr.sel(season="ANN").where(mask_union)
lw_clr_ceres = xr.open_dataset("../pp/ceres_trends.nc").lw_clr.sel(season="ANN").where(mask_union)

sw_cre_ceres = xr.open_dataset("../pp/ceres_trends.nc").sw_cre.sel(season="ANN").where(mask_union)
net_cre_ceres = xr.open_dataset("../pp/ceres_trends.nc").net_cre.sel(season="ANN").where(mask_union)
lw_cre_ceres = xr.open_dataset("../pp/ceres_trends.nc").lw_cre.sel(season="ANN").where(mask_union)

sw_ceres = xr.open_dataset("../pp/ceres_trends.nc").sw.sel(season="ANN").where(mask_union)
net_ceres = xr.open_dataset("../pp/ceres_trends.nc").net.sel(season="ANN").where(mask_union)
lw_ceres = xr.open_dataset("../pp/ceres_trends.nc").lw.sel(season="ANN").where(mask_union)

#Print trends

print('Land SW clear sky trend in the SL:', np.round(sw_clr_ceres.weighted(weights).mean(("lat", "lon")).values,2))
print('Land LW clear sky trend in the SL:', np.round(lw_clr_ceres.weighted(weights).mean(("lat", "lon")).values,2))
print('Land net clear sky trend in the SL:', np.round(net_clr_ceres.weighted(weights).mean(("lat", "lon")).values,2))

print('Land SW cre trend in the SL:', np.round(sw_cre_ceres.weighted(weights).mean(("lat", "lon")).values,2))
print('Land LW cre sky trend in the SL:', np.round(lw_cre_ceres.weighted(weights).mean(("lat", "lon")).values,2))
print('Land net cre sky trend in the SL:', np.round(net_cre_ceres.weighted(weights).mean(("lat", "lon")).values,2))

print('Land SW trend in the SL:', np.round(sw_ceres.weighted(weights).mean(("lat", "lon")).values,2))
print('Land LW trend in the SL:', np.round(lw_ceres.weighted(weights).mean(("lat", "lon")).values,2))
print('Land net trend in the SL:', np.round(net_ceres.weighted(weights).mean(("lat", "lon")).values,2))

#Calculate CERES albedo
ceres_albedo = (ceres.sfc_sw_up_all_mon / ceres.sfc_sw_down_all_mon).to_dataset(name="albedo")
a_trend = trend_and_ci(ceres_albedo, vars = ["albedo"])


#Correlation between albedo trend as SW trend
xr.corr(a_trend.albedo.where(mask_union),sw_clr_ceres.where(mask_union)).values

#get covariability of SW trend with albedo changes
from scipy import stats

# Get the two variables for comparison
x = a_trend.where(mask_union).values.flatten()
y = sw_clr_ceres.where(mask_union).values.flatten()

# Remove NaN values
valid_mask = ~(np.isnan(x) | np.isnan(y))
x_clean = x[valid_mask]
y_clean = y[valid_mask]

# Calculate linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

print(f"Slope: {slope:.6f}")
print(f"Intercept: {intercept:.6f}")
print(f"R-squared: {r_value**2:.6f}")
print(f"P-value: {p_value:.6e}")
print(f"Standard error: {std_err:.6f}")

sw_trend_explained_by_albedo = slope*a_trend.where(mask_union)
sw_trend_residual = sw_clr_ceres - sw_trend_explained_by_albedo

print('total SW clr trend:',np.round(sw_clr_ceres.weighted(weights).mean(("lat", "lon")).values,2),'W / m^2 / dec')
print('SW trend explained by albedo:',np.round(sw_trend_explained_by_albedo.weighted(weights).mean(("lat", "lon")).values,2) ,'W / m^2 / dec')
print('Residual:', np.round(sw_trend_residual.weighted(weights).mean(("lat", "lon")).values,2),'W / m^2 / dec')