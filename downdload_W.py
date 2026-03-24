import xarray as xr
from scipy.ndimage import gaussian_filter


def Downdload_W():
    import cdsapi

    dataset = "reanalysis-era5-pressure-levels-monthly-means"
    years = [str(year) for year in range(2001, 2025)]
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["vertical_velocity"],
        "pressure_level": ["500"],
        "year": years,
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "grid": "1/1",
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()


def seasonal_averages():

    ds = xr.open_dataset("output/W_raw.nc").sortby("latitude")
    ds = ds.rename({"longitude": "lon", "latitude": "lat", "valid_time": "time"})
    ds = ds.groupby("time.season").mean("time", skipna=True).squeeze()

    # Apply 2D Gaussian smoothing to the data variable
    ds["w"] = xr.apply_ufunc(
        gaussian_filter,
        ds["w"],
        kwargs={"sigma": 5},
        dask="parallelized",
        output_dtypes=[ds["w"].dtype],
    )

    ds.to_netcdf("output/W_mean.nc")


if __name__ == "__main__":
    seasonal_averages()
