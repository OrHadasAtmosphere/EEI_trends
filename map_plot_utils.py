from __future__ import annotations

import numpy as np
import xarray as xr


def wrap_global_field(field: xr.DataArray) -> tuple[np.ndarray, xr.DataArray, np.ndarray]:
    lon = field["lon"].data
    lon = np.concatenate([lon, lon[:1] + 360.0])
    data = np.asarray(field.data)
    data = np.concatenate([data, data[..., :1]], axis=-1)
    return lon, field["lat"], data
