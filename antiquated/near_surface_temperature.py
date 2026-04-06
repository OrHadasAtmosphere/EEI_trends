from __future__ import annotations

import os
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib.pyplot as plt


INPUT_FILE = SCRIPT_DIR / "output" / "near_sruface_termperature.nc"
ICE_FILE = SCRIPT_DIR / "output" / "ice_mean.nc"
OUTPUT_FILE = SCRIPT_DIR / "output" / "near_surface_temperature_over_sea_ice.nc"
FIGURE_FILE = SCRIPT_DIR / "figures" / "near_surface_temperature.png"
SEA_ICE_THRESHOLD = 0.15
EARTH_RADIUS_M = 6_371_000.0


def download(target: Path = INPUT_FILE) -> Path:
    years = [str(year) for year in range(2000, 2025)]
    dataset = "reanalysis-era5-single-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_dewpoint_temperature", "2m_temperature"],
        "year": years,
        "month": ["01", "02", "12"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "grid": "1/1",
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(str(target))
    return target


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


def load_near_surface_fields(path: Path = INPUT_FILE) -> xr.Dataset:
    ds = xr.open_dataset(path).sortby("latitude")
    rename_map = {
        "valid_time": "time",
        "latitude": "lat",
        "longitude": "lon",
    }
    ds = ds.rename({key: value for key, value in rename_map.items() if key in ds.dims or key in ds.coords})

    if "number" in ds.dims:
        ds = ds.squeeze("number", drop=True)

    return ds[["t2m", "d2m"]].sortby("lat")


def load_mean_sh_djf_sea_ice(
    path: Path = ICE_FILE,
    threshold: float = SEA_ICE_THRESHOLD,
) -> tuple[xr.DataArray, xr.DataArray]:
    ds = xr.open_dataset(path).sortby("lat")
    if "number" in ds.dims:
        ds = ds.squeeze("number", drop=True)

    djf_ice = ds["siconc"].sel(season="DJF").sortby("lat")
    djf_ice_sh = djf_ice.where(djf_ice["lat"] < 0.0)
    sea_ice_mask = (djf_ice_sh >= threshold).fillna(False)
    sea_ice_mask.name = "mean_djf_sh_sea_ice_extent_mask"
    sea_ice_mask.attrs.update(
        long_name="Southern Hemisphere mean DJF sea-ice extent mask",
        description=(
            "Mask derived from the DJF mean sea-ice concentration in output/ice_mean.nc. "
            f"Sea ice extent is defined where sea-ice concentration is at least {threshold:.2f}."
        ),
        units="1",
        threshold=threshold,
    )
    return djf_ice_sh, sea_ice_mask


def latitude_weights(lat: xr.DataArray) -> xr.DataArray:
    weights = xr.DataArray(
        np.cos(np.deg2rad(lat.values)),
        coords={"lat": lat},
        dims=("lat",),
        name="latitude_weight",
    )
    return weights


def cell_area(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    if lat.size < 2 or lon.size < 2:
        raise ValueError("Need at least two latitude and longitude points to estimate grid-cell area.")

    dlat = np.deg2rad(float(np.abs(lat.diff("lat").median())))
    dlon = np.deg2rad(float(np.abs(lon.diff("lon").median())))
    lat_radians = np.deg2rad(lat.values)
    area_1d = (EARTH_RADIUS_M ** 2) * dlat * dlon * np.cos(lat_radians)

    return xr.DataArray(
        area_1d[:, None] * np.ones(lon.size, dtype=np.float64),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="cell_area",
        attrs={"units": "m2", "long_name": "Approximate grid-cell area"},
    )


def area_weighted_mean_over_mask(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    mask_on_grid = mask.sel(lat=da["lat"], lon=da["lon"])
    weights = latitude_weights(da["lat"]).broadcast_like(mask_on_grid)
    masked_weights = weights.where(mask_on_grid)

    numerator = (da.where(mask_on_grid) * masked_weights).sum(("lat", "lon"), skipna=True)
    denominator = masked_weights.where(np.isfinite(da)).sum(("lat", "lon"), skipna=True)
    return numerator / denominator


def linear_trend(series: xr.DataArray) -> tuple[float, np.ndarray]:
    years = series["year"].values.astype(np.float64)
    values = series.values.astype(np.float64)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.nan, np.full(values.shape, np.nan, dtype=np.float64)

    slope, intercept = np.polyfit(years[valid], values[valid], 1)
    fitted = intercept + slope * years
    return float(slope), fitted


def build_djf_timeseries_dataset(
    input_file: Path = INPUT_FILE,
    ice_file: Path = ICE_FILE,
    output_file: Path = OUTPUT_FILE,
    threshold: float = SEA_ICE_THRESHOLD,
) -> xr.Dataset:
    fields = load_near_surface_fields(input_file)
    djf_t2m = seasonal_means(fields["t2m"], "DJF")
    djf_d2m = seasonal_means(fields["d2m"], "DJF")

    mean_djf_ice, sea_ice_mask = load_mean_sh_djf_sea_ice(ice_file, threshold=threshold)
    sea_ice_mask = sea_ice_mask.sel(lat=djf_t2m["lat"], lon=djf_t2m["lon"])
    mean_djf_ice = mean_djf_ice.sel(lat=djf_t2m["lat"], lon=djf_t2m["lon"])

    t2m_series = area_weighted_mean_over_mask(djf_t2m, sea_ice_mask)
    d2m_series = area_weighted_mean_over_mask(djf_d2m, sea_ice_mask)

    t2m_series.name = "t2m_over_mean_djf_sh_sea_ice"
    t2m_series.attrs.update(
        long_name="DJF mean 2 metre temperature over mean Southern Hemisphere sea-ice extent",
        units="K",
    )
    d2m_series.name = "d2m_over_mean_djf_sh_sea_ice"
    d2m_series.attrs.update(
        long_name="DJF mean 2 metre dewpoint temperature over mean Southern Hemisphere sea-ice extent",
        units="K",
    )

    area = cell_area(djf_t2m["lat"], djf_t2m["lon"])
    extent_area_m2 = area.where(sea_ice_mask).sum(("lat", "lon"), skipna=True)
    extent_area_million_km2 = float(extent_area_m2.values) / 1.0e12

    t2m_slope, _ = linear_trend(t2m_series)
    d2m_slope, _ = linear_trend(d2m_series)

    out = xr.Dataset(
        data_vars={
            "t2m_over_mean_djf_sh_sea_ice": t2m_series.astype(np.float32),
            "d2m_over_mean_djf_sh_sea_ice": d2m_series.astype(np.float32),
            "mean_djf_sh_sea_ice_concentration": mean_djf_ice.astype(np.float32),
            "mean_djf_sh_sea_ice_extent_mask": sea_ice_mask.astype(np.int8),
            "mean_djf_sh_sea_ice_extent_million_km2": xr.DataArray(
                np.float32(extent_area_million_km2),
                attrs={
                    "long_name": "Mean Southern Hemisphere DJF sea-ice extent",
                    "description": (
                        "Extent derived from the DJF climatological sea-ice concentration "
                        f"using a {threshold:.2f} concentration threshold."
                    ),
                    "units": "10^6 km2",
                },
            ),
            "t2m_trend_over_mean_djf_sh_sea_ice": xr.DataArray(
                np.float32(t2m_slope * 10.0),
                attrs={
                    "long_name": "Linear trend of DJF mean 2 metre temperature over mean SH sea-ice extent",
                    "units": "K decade-1",
                },
            ),
            "d2m_trend_over_mean_djf_sh_sea_ice": xr.DataArray(
                np.float32(d2m_slope * 10.0),
                attrs={
                    "long_name": "Linear trend of DJF mean 2 metre dewpoint temperature over mean SH sea-ice extent",
                    "units": "K decade-1",
                },
            ),
        },
        coords={"year": djf_t2m["year"], "lat": djf_t2m["lat"], "lon": djf_t2m["lon"]},
        attrs={
            "title": "DJF near-surface temperature and dewpoint over mean Southern Hemisphere sea-ice extent",
            "source_temperature_file": str(input_file),
            "source_ice_file": str(ice_file),
            "sea_ice_extent_threshold": threshold,
            "history": "Created by basic_trend/near_surface_temperature.py",
        },
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        "t2m_over_mean_djf_sh_sea_ice": {"dtype": "float32"},
        "d2m_over_mean_djf_sh_sea_ice": {"dtype": "float32"},
        "mean_djf_sh_sea_ice_concentration": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "mean_djf_sh_sea_ice_extent_mask": {"zlib": True, "complevel": 4, "dtype": "int8"},
        "mean_djf_sh_sea_ice_extent_million_km2": {"dtype": "float32"},
        "t2m_trend_over_mean_djf_sh_sea_ice": {"dtype": "float32"},
        "d2m_trend_over_mean_djf_sh_sea_ice": {"dtype": "float32"},
    }
    out.to_netcdf(output_file, encoding=encoding)
    return out


def plot_djf_timeseries(
    ds: xr.Dataset,
    figure_file: Path = FIGURE_FILE,
) -> Path:
    years = ds["year"].values.astype(int)
    t2m_c = ds["t2m_over_mean_djf_sh_sea_ice"].values - 273.15
    d2m_c = ds["d2m_over_mean_djf_sh_sea_ice"].values - 273.15

    t2m_slope_decade, t2m_fit = linear_trend(ds["t2m_over_mean_djf_sh_sea_ice"])
    d2m_slope_decade, d2m_fit = linear_trend(ds["d2m_over_mean_djf_sh_sea_ice"])

    figure_file.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    ax.plot(years, t2m_c, color="firebrick", marker="o", linewidth=2, label="2 m temperature")
    ax.plot(years, d2m_c, color="steelblue", marker="s", linewidth=2, label="2 m dew point")

    ax.plot(
        years,
        t2m_fit - 273.15,
        color="firebrick",
        linestyle="--",
        linewidth=1.5,
        label=f"Temperature trend: {t2m_slope_decade * 10.0:.2f} C decade-1",
    )
    ax.plot(
        years,
        d2m_fit - 273.15,
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        label=f"Dew point trend: {d2m_slope_decade * 10.0:.2f} C decade-1",
    )

    extent = float(ds["mean_djf_sh_sea_ice_extent_million_km2"].values)
    ax.set_title(
        "DJF Near-Surface Temperature and Dew Point Over Mean SH Sea-Ice Extent\n"
        f"Sea-ice extent threshold >= {SEA_ICE_THRESHOLD:.2f}, extent = {extent:.2f} x10^6 km2"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (C)")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, ncol=2)

    fig.savefig(figure_file, dpi=300)
    plt.close(fig)
    return figure_file


def main() -> xr.Dataset:
    ds = build_djf_timeseries_dataset()
    plot_djf_timeseries(ds)
    return ds


if __name__ == "__main__":
    main()
