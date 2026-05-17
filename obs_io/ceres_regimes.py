from . import read_ceres, regime_trend

ds = read_ceres()
slices = [slice(None, None), slice("2005", None), slice(None, "2021")]
fname_ext_per_slice = [".nc", "_using_ceres_from_2005.nc", "_using_ceres_until_2021.nc"]
for slc, nm_ext in zip(slices, fname_ext_per_slice):
    parent_dir = "diff_yrs/" if nm_ext != ".nc" else ""
    print(f"calculating regime trends for sliced CERES years: {slc.start}-{slc.stop}")
    regime_trend(ds.sel(year=slc), "regime_masks.nc", f"{parent_dir}regime_mean_trends{nm_ext}")

print("\ncalculating regime-mean trends using different climatologies to define regimes:")

for fmask, fout in zip(
    ["diff_yrs/regime_masks_1995.nc", "diff_yrs/regime_masks_2000.nc"],
    ["diff_yrs/regime_mean_trends_1995.nc", "diff_yrs/regime_mean_trends_2000.nc"]
):
    regime_trend(ds, fmask, fout)
    