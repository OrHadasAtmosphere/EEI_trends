from __future__ import annotations

import matplotlib as mpl
import numpy as np
import xarray as xr

GLOBAL_FONT_SIZE = 12
DEFAULT_COLORBAR_THICKNESS = 0.04

mpl.rcParams.update({"font.size": GLOBAL_FONT_SIZE})


def wrap_global_field(
    field: xr.DataArray,
) -> tuple[np.ndarray, xr.DataArray, np.ndarray]:
    lon = field["lon"].data
    lon = np.concatenate([lon, lon[:1] + 360.0])
    data = np.asarray(field.data)
    data = np.concatenate([data, data[..., :1]], axis=-1)
    return lon, field["lat"], data


def add_colorbar(
    fig,
    mappable,
    *,
    ax,
    orientation: str = "horizontal",
    thickness: float | None = None,
    **kwargs,
):
    if thickness is not None:
        kwargs.setdefault("fraction", thickness)
    return fig.colorbar(
        mappable,
        ax=ax,
        orientation=orientation,
        **kwargs,
    )


#try