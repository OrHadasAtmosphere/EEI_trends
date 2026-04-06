from __future__ import annotations

import numpy as np
import scipy.stats as stats
import xarray as xr


def fit_trend_map(
    da: xr.DataArray,
    time_dim: str = "year",
    min_count: int = 2,
    significance_level: float = 0.05,
) -> xr.Dataset:
    spatial_dims = [dim for dim in da.dims if dim != time_dim]
    ordered = da.transpose(time_dim, *spatial_dims).astype(np.float64)

    x = ordered[time_dim].values.astype(np.float64)
    y = ordered.values.reshape(ordered.sizes[time_dim], -1)
    mask = np.isfinite(y)
    x2d = x[:, None]
    valid_counts = mask.sum(axis=0)

    enough_points = valid_counts >= min_count
    x_sum = np.where(mask, x2d, 0.0).sum(axis=0)
    y_sum = np.where(mask, y, 0.0).sum(axis=0)

    x_mean = np.full(y.shape[1], np.nan, dtype=np.float64)
    y_mean = np.full(y.shape[1], np.nan, dtype=np.float64)
    x_mean[enough_points] = x_sum[enough_points] / valid_counts[enough_points]
    y_mean[enough_points] = y_sum[enough_points] / valid_counts[enough_points]

    x_centered = x2d - x_mean
    y_centered = y - y_mean
    sxx = np.where(mask, x_centered**2, 0.0).sum(axis=0)
    sxy = np.where(mask, x_centered * y_centered, 0.0).sum(axis=0)

    slope = np.full(y.shape[1], np.nan, dtype=np.float32)
    valid = enough_points & (sxx > 0.0)
    slope[valid] = (sxy[valid] / sxx[valid]).astype(np.float32)

    intercept = y_mean - slope.astype(np.float64) * x_mean
    fitted = intercept + slope.astype(np.float64) * x2d
    residual = np.where(mask, y - fitted, 0.0)
    sse = (residual**2).sum(axis=0)
    dof = valid_counts - 2

    mse = np.full(y.shape[1], np.nan, dtype=np.float64)
    slope_se = np.full(y.shape[1], np.nan, dtype=np.float64)
    testable = valid & (dof > 0)
    mse[testable] = sse[testable] / dof[testable]
    slope_se[testable] = np.sqrt(mse[testable] / sxx[testable])

    t_stat = np.full(y.shape[1], np.nan, dtype=np.float64)
    nonzero_se = testable & (slope_se > 0.0)
    t_stat[nonzero_se] = slope[nonzero_se] / slope_se[nonzero_se]

    p_value = np.full(y.shape[1], np.nan, dtype=np.float32)
    p_value[nonzero_se] = (
        2.0 * (1.0 - stats.t.cdf(np.abs(t_stat[nonzero_se]), dof[nonzero_se]))
    ).astype(np.float32)

    significant = np.zeros(y.shape[1], dtype=np.int8)
    significant[nonzero_se] = (
        p_value[nonzero_se] < significance_level
    ).astype(np.int8)

    coords = {dim: ordered[dim] for dim in spatial_dims}
    shape = tuple(ordered.sizes[dim] for dim in spatial_dims)
    return xr.Dataset(
        data_vars={
            "slope": (spatial_dims, slope.reshape(*shape)),
            "p_value": (spatial_dims, p_value.reshape(*shape)),
            "significant": (spatial_dims, significant.reshape(*shape)),
            "count": (
                spatial_dims,
                valid_counts.astype(np.int16).reshape(*shape),
            ),
        },
        coords=coords,
    )
