import numpy as np
import xarray as xr
import glob
import datetime

days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
cumdays_in_month = np.cumsum(days_in_month)

## open raw hdf files and merge into xarray dataset
for sat_name, sat_prefix in [("terra","MOD"), ("aqua","MYD")]:
    files = glob.glob("raw/**/MOD*.hdf", recursive=True)
    print(len(files))

    for i,f in enumerate(files):
        date = f.split(".A")[1].split(".")[0]
        year, day = date[:4], date[4:]
        for j,c in enumerate(cumdays_in_month):
            if int(day) < c:
                time = datetime.datetime(year=int(year), month=j+1, day=15)
                break
        ds_orig = xr.open_dataset(f, engine="netcdf4")
        dsi = xr.Dataset(
            data_vars = {
                "aod":(("lat","lon","time"), ds_orig.Aerosol_Optical_Depth_Land_Ocean_Mean_Mean.values[:,:, np.newaxis]), 
            }, 
            coords = {
                "lat":ds_orig.YDim.values, 
                "lon":ds_orig.XDim.values,
                "time":[time],
            }
        )
        if i > 0:
            ds = ds.merge(dsi)
        else:
            ds = dsi
    ds_orig.close()
    dsi.close()
    ds = ds.sortby("time")
    ds["lon"] = (ds.coords['lon'] + 360) % 360
    ds = ds.sortby("lon").sortby("lat")    
    ds.to_netcdf("./"+sat_name+"_combined.nc")
    ds.close()

## merge together aqua and terra
terra = xr.open_dataset("terra_combined.nc").sortby("lat")
terra["lon"] = (terra.coords['lon'] + 360) % 360
terra = terra.sortby("lon").drop_vars("sza")
terra = terra.rename_vars({"aod":"aod_terra"})
aqua = xr.open_dataset("aqua_combined.nc").sortby("lat")
aqua["lon"] = (aqua.coords['lon'] + 360) % 360
aqua = aqua.sortby("lon").drop_vars("sza")
aqua = aqua.rename_vars({"aod":"aod_aqua"})
ds = xr.merge([terra, aqua])
terra.close()
aqua.close()

aod_modis = np.nanmean(np.stack([ds.aod_aqua.values, ds.aod_terra.values]), axis=0)
aod_modis = xr.DataArray(aod_modis, coords=ds.aod_aqua.coords, dims=ds.aod_aqua.dims)
ds = aod_modis.to_dataset(name="aod")
ds = ds.sel(time=slice("2000-03-01", "2026-03-01"))
ds.to_netcdf("modis_aod.nc")
print("done")
