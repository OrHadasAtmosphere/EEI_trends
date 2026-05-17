from . import read_ceres, to_trend, regimes_from_clim, regime_trend

ds = read_ceres()

slices = [slice("2005", None), slice(None, "2021")]

trendfiles = ["diff_yrs/ceres_trends_from_2005.nc", "diff_yrs/ceres_trends_until_2021.nc"]

for slc, tfile in zip(slices, trendfiles):
    # save linear trends
    to_trend(ds.sel(year=slc)).to_netcdf(f"pp/{tfile}")
    print(f"done linear trend: with CERES record cut using subselection {slc.start}-{slc.stop}")

# define regimes from different reference periods
infile = ["diff_yrs/era5_clim_1995.nc", "diff_yrs/era5_clim_2000.nc"]
outfile = ["diff_yrs/regime_masks_1995.nc", "diff_yrs/regime_masks_2000.nc"]

for fin, fout in zip(infile, outfile):
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