from . import read_ceres, to_trend, regimes_from_clim, regime_trend, process_era5_clim
from .process_era5clim import seasonal_clim

ds = read_ceres()

slices = [slice("2005", None), slice(None, "2021")]

trendfiles = ["diff_yrs/ceres_trends_from_2005.nc", "diff_yrs/ceres_trends_until_2021.nc"]

for slc, tfile in zip(slices, trendfiles):
    # save linear trends
    to_trend(ds.sel(year=slc)).to_netcdf(f"pp/{tfile}")
    print(f"done linear trend: with CERES record cut using subselection {slc.start}-{slc.stop}")

####
# get seasonal climatology for ERA5 periods:
####
ds_full = process_era5_clim()
for sl, out_ext in zip(
    [slice("1995-03-01", "2005-03-01"), slice("2000-03-01", "2010-03-01")],
    ["_1995", "_2000"]
):
    seasonal_clim(ds_full, sl, out_ext)

# define regimes from different reference periods
infile = ["diff_yrs/era5_clim_1995.nc", "diff_yrs/era5_clim_2000.nc"]
outfile = ["diff_yrs/regime_masks_1995.nc", "diff_yrs/regime_masks_2000.nc"]

for fin, fout in zip(infile, outfile):
    print(f"defining regime masks from climatology in {infile}")
    regimes_from_clim(fin, fout)

# calculate regime trends from sliced CERES record and from shifted ERA5 climatologies

print("\ncalculating regime trends from sliced CERES record")
fname_ext_per_slice = ["_using_ceres_from_2005.nc", "_using_ceres_until_2021.nc"]
for slc, nm_ext in zip(slices, fname_ext_per_slice):
    print(f"calculating regime trends for sliced CERES years: {slc.start}-{slc.stop}")
    regime_trend(ds.sel(year=slc), "regime_masks.nc", f"diff_yrs/regime_mean_trends{nm_ext}")

print("\ncalculating regime-mean trends using different climatologies to define regimes:")
for fmask, fout in zip(
    ["diff_yrs/regime_masks_1995.nc", "diff_yrs/regime_masks_2000.nc"],
    ["diff_yrs/regime_mean_trends_1995.nc", "diff_yrs/regime_mean_trends_2000.nc"]
):
    regime_trend(ds, fmask, fout)