import cdsapi

target = 'raw_data/era5_land_sea_mask.nc'
dataset = 'reanalysis-era5-single-levels-monthly-means'
request = {
    'product_type': 'monthly_averaged_reanalysis',
    'variable': 'land_sea_mask',
    'year': '2020',
    'month': '01',
    'day': '01',
    'time': '00:00',
    'format': 'netcdf',
    'grid': [1, 1],
    'download_format': 'unarchived',
}

client = cdsapi.Client()
client.retrieve(dataset, request, target)