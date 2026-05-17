from . import read_ceres, to_trend
from utils import global_mean, trend_and_ci

ds = read_ceres()

# save linear trends
to_trend(ds).to_netcdf(f"pp/ceres_trends.nc")
print("done linear trend")

# save global-mean
ds_gm = global_mean(ds).load()
gm_trend = trend_and_ci(ds_gm)
gm_trend.to_netcdf("pp/ceres_gm_timeseries.nc")
ds_gm.close()
gm_trend.close()
print("done global-mean")