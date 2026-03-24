from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


PERIODS = ("annual", "DJF", "MAM", "JJA", "SON")
SEASON_LABELS = {
    "all_sky": "all-sky",
    "clear_sky": "clear-sky",
    "all_clear": "all-sky minus clear-sky",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate CERES EBAF TOA trend maps for annual means and seasonal means "
            "and save them as a NetCDF file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/orhadas/Downloads/CERES_EBAF-TOA_Ed4.2.1_Subset_200003-202512.nc"),
        help="Input CERES NetCDF file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("basic_trend/output/ceres_ebaf_toa_trend_maps.nc"),
        help="Output NetCDF file.",
    )
    return parser.parse_args()


def ensure_required_variables(ds: xr.Dataset) -> None:
    required = {
        "toa_sw_all_mon",
        "toa_lw_all_mon",
        "toa_sw_clr_c_mon",
        "toa_lw_clr_c_mon",
    }
    missing = sorted(required.difference(ds.data_vars))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Input file is missing required CERES variables: {joined}")


def group_complete_means(
    da: xr.DataArray,
    labels: xr.DataArray,
    expected_count: int,
    output_dim: str,
) -> xr.DataArray:
    grouped = da.groupby(labels).mean("time", skipna=True)
    grouped = grouped.rename({labels.name: output_dim})

    counts = labels.groupby(labels).count()
    counts = counts.rename({labels.name: output_dim})
    valid_groups = counts.where(counts == expected_count, drop=True)[output_dim]

    return grouped.sel({output_dim: valid_groups})


def annual_means(da: xr.DataArray) -> xr.DataArray:
    labels = xr.DataArray(
        da["time"].dt.year.values,
        coords={"time": da["time"]},
        dims=("time",),
        name="year",
    )
    return group_complete_means(da, labels, expected_count=12, output_dim="year")


def seasonal_means(da: xr.DataArray, season: str) -> xr.DataArray:
    season_mask = da["time"].dt.season == season
    season_da = da.where(season_mask, drop=True)

    season_year = season_da["time"].dt.year.values.astype(np.int32)
    december = season_da["time"].dt.month.values == 12
    season_year = season_year + december.astype(np.int32)

    labels = xr.DataArray(
        season_year,
        coords={"time": season_da["time"]},
        dims=("time",),
        name="season_year",
    )
    return group_complete_means(season_da, labels, expected_count=3, output_dim="year")


def compute_trend_map(da: xr.DataArray, time_dim: str = "year") -> xr.DataArray:
    spatial_dims = [dim for dim in da.dims if dim != time_dim]
    ordered = da.transpose(time_dim, *spatial_dims).astype(np.float64)

    x = ordered[time_dim].values.astype(np.float64)
    y = ordered.values.reshape(ordered.sizes[time_dim], -1)

    mask = np.isfinite(y)
    x2d = x[:, None]
    valid_counts = mask.sum(axis=0)

    y_sum = np.where(mask, y, 0.0).sum(axis=0)
    x_sum = np.where(mask, x2d, 0.0).sum(axis=0)

    x_mean = np.full(y.shape[1], np.nan, dtype=np.float64)
    y_mean = np.full(y.shape[1], np.nan, dtype=np.float64)
    enough_points = valid_counts >= 2
    x_mean[enough_points] = x_sum[enough_points] / valid_counts[enough_points]
    y_mean[enough_points] = y_sum[enough_points] / valid_counts[enough_points]

    covariance = np.where(mask, (x2d - x_mean) * (y - y_mean), 0.0).sum(axis=0)
    variance = np.where(mask, (x2d - x_mean) ** 2, 0.0).sum(axis=0)

    slope = np.full(y.shape[1], np.nan, dtype=np.float32)
    valid = enough_points & (variance > 0)
    slope[valid] = (covariance[valid] / variance[valid]).astype(np.float32)

    coords = {dim: ordered[dim] for dim in spatial_dims}
    return xr.DataArray(
        slope.reshape(*(ordered.sizes[dim] for dim in spatial_dims)),
        coords=coords,
        dims=spatial_dims,
    )


def build_period_trends(da: xr.DataArray) -> tuple[xr.DataArray, list[int]]:
    trend_maps: list[xr.DataArray] = []
    sample_counts: list[int] = []

    for period in PERIODS:
        if period == "annual":
            averaged = annual_means(da)
        else:
            averaged = seasonal_means(da, period)

        trend_map = compute_trend_map(averaged, time_dim="year").expand_dims(period=[period])
        trend_maps.append(trend_map)
        sample_counts.append(int(averaged.sizes["year"]))

    return xr.concat(trend_maps, dim="period"), sample_counts


def build_output_dataset(ds: xr.Dataset) -> xr.Dataset:
    fields = {
        "all_sky": {
            "shortwave": ds["toa_sw_all_mon"],
            "longwave": ds["toa_lw_all_mon"],
        },
        "clear_sky": {
            "shortwave": ds["toa_sw_clr_c_mon"],
            "longwave": ds["toa_lw_clr_c_mon"],
        },
    }

    fields["all_clear"] = {
        "shortwave": fields["all_sky"]["shortwave"] - fields["clear_sky"]["shortwave"],
        "longwave": fields["all_sky"]["longwave"] - fields["clear_sky"]["longwave"],
    }

    for sky_name, components in fields.items():
        components["net"] = -(components["shortwave"] + components["longwave"])

    out = xr.Dataset(coords={"period": list(PERIODS), "lat": ds["lat"], "lon": ds["lon"]})
    sample_counts: list[int] | None = None

    for sky_name, components in fields.items():
        for component_name, da in components.items():
            trends, current_counts = build_period_trends(da)
            if sample_counts is None:
                sample_counts = current_counts

            var_name = f"{sky_name}_{component_name}_trend"
            long_name = f"{SEASON_LABELS[sky_name]} {component_name} trend"
            description = (
                f"Linear trend from complete {component_name} annual/seasonal means for "
                f"{SEASON_LABELS[sky_name]} conditions."
            )
            if component_name == "net":
                description += (
                    " Net is derived as the negative sum of outgoing shortwave and "
                    "outgoing longwave fluxes, so positive trends indicate increasing "
                    "downward TOA net flux."
                )
            if sky_name == "all_clear":
                description += " The all_clear fields are defined as all-sky minus clear-sky."

            trends = trends.astype(np.float32)
            trends.attrs.update(
                long_name=long_name,
                units="W m-2 yr-1",
                description=description,
            )
            out[var_name] = trends

    out["sample_count"] = xr.DataArray(
        np.asarray(sample_counts, dtype=np.int16),
        coords={"period": list(PERIODS)},
        dims=("period",),
        attrs={
            "long_name": "Number of complete annual or seasonal means used in each trend fit",
            "units": "1",
        },
    )

    out.attrs.update(
        title="CERES EBAF TOA annual and seasonal trend maps",
        source_file=str(ds.encoding.get("source", "")),
        history="Created by basic_trend/ceres_trend_maps.py",
        note=(
            "Input subset contains outgoing shortwave and longwave fluxes for all-sky and "
            "clear-sky conditions. all_clear = all-sky minus clear-sky."
        ),
    )
    return out


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.input, engine="netcdf4")
    ensure_required_variables(ds)
    out = build_output_dataset(ds)

    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in out.data_vars
        if name != "sample_count"
    }
    encoding["sample_count"] = {"dtype": "int16"}

    out.to_netcdf(args.output, engine="netcdf4", encoding=encoding)
    print(args.output)


if __name__ == "__main__":
    main()
