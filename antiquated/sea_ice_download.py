import cdsapi
import xarray as xr
def download():
    years = [str(year) for year in range(2001, 2025)]
    dataset = "reanalysis-era5-single-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["sea_ice_cover"],
        "year": years,
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
                "grid": "1/1",
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()

def seasonal_averages():

    ds = xr.open_dataset("output/ice_raw.nc").sortby("latitude")
    ds = ds.rename({"longitude": "lon", "latitude": "lat", "valid_time": "time"})
    ds = ds.groupby("time.season").mean("time", skipna=True).squeeze()
    ds.to_netcdf("output/ice_mean.nc")


if __name__ == "__main__":
    seasonal_averages()
