from . import read_ceres, regime_trend

ds = read_ceres()
regime_trend(ds, "regime_masks.nc", "regime_mean_trends.nc")
    