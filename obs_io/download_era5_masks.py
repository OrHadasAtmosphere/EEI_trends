import cdsapi

dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["land_sea_mask", "sea_ice_cover"],
    "year": [str(year) for year in range(1986, 2011)],
    "month": [f"{month:02d}" for month in range(1, 13)],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "grid": [1, 1],
}


def download_era5_masks(target="raw_data/masks_levels.nc"):
    client = cdsapi.Client()
    client.retrieve(dataset, request, target)


if __name__ == "__main__":
    download_era5_masks()
