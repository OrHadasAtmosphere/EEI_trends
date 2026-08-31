from . import to_trend
from .ceres import read_ceres_raw
from .ceres_regimes import regime_trend
from .process_masks import regimes_from_clim
from .process_era5clim import process_era5_clim, seasonal_clim

ds_ceres = read_ceres_raw()

ceres_slices = [slice("2005", None), slice(None, "2021")]
trendfiles = ["diff_yrs/ceres_trends_from_2005.nc", "diff_yrs/ceres_trends_until_2021.nc"]
for slc, tfile in zip(ceres_slices, trendfiles):
    # save linear trends
    to_trend(ds_ceres.sel(year=slc)).to_netcdf(f"pp/{tfile}")
    print(f"done linear trend: with CERES record cut using subselection {slc.start}-{slc.stop}")

####
# get seasonal climatology for ERA5 periods:
####
ds_era = process_era5_clim()
slices = [slice("1995-03-01", "2005-03-01"),
     slice("2000-03-01", "2010-03-01"),
     slice("2005-03-01", "2015-03-01"),
     slice("2010-03-01", "2020-03-01"),
     slice("2015-03-01", "2025-03-01"),
    ]
file_extensions = ["_1995", "_2000", "_2005", "_2010", "_2015",]

print("make climatologies over different years")
for sl, out_ext in zip(slices, file_extensions):
    print(f"making clim for {out_ext}")
    seasonal_clim(ds_era, sl, "diff_yrs/", out_ext)

# define regimes from different reference periods
for ext in file_extensions:
    fin = f"diff_yrs/era5_clim{ext}.nc"
    fout = f"diff_yrs/regime_masks{ext}.nc"
    print(f"defining regime masks from climatology in {fin}")
    regimes_from_clim(fin, fout)

# calculate regime trends from sliced CERES record and from shifted ERA5 climatologies
print("\ncalculating regime trends from sliced CERES record")
fname_ext_per_slice = ["_using_ceres_from_2005.nc", "_using_ceres_until_2021.nc"]
for slc, nm_ext in zip(ceres_slices, fname_ext_per_slice):
    print(f"calculating regime trends for sliced CERES years: {slc.start}-{slc.stop}")
    regime_trend(ds_ceres.sel(year=slc), "regime_masks.nc", f"diff_yrs/regime_mean_trends{nm_ext}")

print("\ncalculating regime-mean trends using different climatologies to define regimes:")
for ext in file_extensions:
    fmask = f"diff_yrs/regime_masks{ext}.nc"
    fout = f"diff_yrs/regime_mean_trends{ext}.nc"
    regime_trend(ds_ceres, fmask, fout)