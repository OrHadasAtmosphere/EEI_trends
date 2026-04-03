import cdsapi

target = "raw_data/masks_levels.nc"
dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "land_sea_mask",
        "sea_ice_cover"
    ],
    "year": [
        "1990", "1992", "1993",
        "1995", "1996", "1997",
        "1998", "1999", "2000"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "grid": [1, 1],
}

client = cdsapi.Client()
client.retrieve(dataset, request, target).download()

###

target = "raw_data/masks_pressures.nc"
dataset = "reanalysis-era5-pressure-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["vertical_velocity"],
    "pressure_level": ["500"],
    "year": [
        "1990", "1991", "1992",
        "1994", "1995", "1996",
        "1997", "1998", "1999",
        "2000"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request, target).download()
