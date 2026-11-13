from .barchart import do_barchart
from .trends_per_regime import do_regime_trend_barchart

ceres_periods = ["using_ceres_from_2005", "using_ceres_until_2021"]
regime_clim_periods = ["1995", "2000", "2005", "2010", "2015"]

for yr_chng in ceres_periods + regime_clim_periods:
    do_barchart(f"pp/diff_yrs/regime_mean_trends_{yr_chng}.nc", extra_fname=f"_{yr_chng}", out_subdir="diff_yrs/")

for yr_chng in ["_using_ceres_from_2005", "_using_ceres_until_2021"]:
    do_regime_trend_barchart(f"pp/diff_yrs/regime_mean_trends{yr_chng}.nc", extra_fname=yr_chng, out_subdir="diff_yrs/")