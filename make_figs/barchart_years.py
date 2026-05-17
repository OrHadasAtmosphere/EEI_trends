import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils.plotting import colors, hatches

from .barchart import do_barchart

ceres_periods = ["using_ceres_from_2005", "using_ceres_until_2021"]
regime_clim_periods = ["1995", "2000"]

for yr_chng in ceres_periods + regime_clim_periods:
    do_barchart(f"pp/diff_yrs/regime_mean_trends_{yr_chng}.nc", extra_fname=f"_{yr_chng}", out_subdir="diff_yrs/")