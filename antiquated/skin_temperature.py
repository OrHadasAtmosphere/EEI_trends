import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import cdsapi
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from map_plot_utils import wrap_global_field
from trend_utils import fit_trend_map


SEASONS = ("DJF", "MAM", "JJA", "SON")
INPUT_FILE = SCRIPT_DIR / "output" / "skin_raw.nc"
OUTPUT_FILE = SCRIPT_DIR / "output" / "skin_trend.nc"
ANNUAL_FIGURE_FILE = SCRIPT_DIR / "figures" / "skin_trend.png"
DJF_FIGURE_FILE = SCRIPT_DIR / "figures" / "skin_trend_DJF.png"
LATITUDE_LINES = np.arange(-75, 76, 15)
CENTRAL_LONGITUDE = 0


def format_latitude_label(latitude: float) -> str:
    if latitude > 0:
        return f"{int(abs(latitude))}°N"
    if latitude < 0:
        return f"{int(abs(latitude))}°S"
    return "0°"


def add_latitude_lines(ax) -> None:
    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.6,
        color="0.35",
        alpha=0.7,
        linestyle=":",
    )
    gridlines.xlocator = mticker.FixedLocator([])
    gridlines.ylocator = mticker.FixedLocator(LATITUDE_LINES)

    for latitude in LATITUDE_LINES:
        ax.text(
            -177.5,
            float(latitude),
            format_latitude_label(float(latitude)),
            transform=ccrs.PlateCarree(),
            ha="left",
            va="center",
            fontsize=9,
            color="0.25",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.2},
            clip_on=False,
        )


def download(target_file: Path = INPUT_FILE):
    years = [str(year) for year in range(2001, 2025)]
    dataset = "reanalysis-era5-single-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["skin_temperature"],
        "year": years,
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "grid": "1/1",
    }

    target_file.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(str(target_file))


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


def load_monthly_skin(path: Path = INPUT_FILE) -> xr.DataArray:
    ds = xr.open_dataset(path).sortby("latitude")
    rename_map = {
        "valid_time": "time",
        "latitude": "lat",
        "longitude": "lon",
    }
    ds = ds.rename(
        {k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords}
    )

    if "number" in ds.dims:
        ds = ds.squeeze("number", drop=True)

    return ds["skt"].sortby("lat")


def annual_mean_trend(
    input_file: Path = INPUT_FILE,
) -> tuple[xr.DataArray, xr.DataArray]:
    skin = load_monthly_skin(input_file)
    yearly_mean = annual_means(skin)
    trend = fit_trend_map(yearly_mean)["slope"]
    trend.name = "annual_skin_trend"
    trend.attrs.update(
        long_name="Annual mean sea surface temperature trend",
        units="K yr-1",
        description=(
            "Linear trend from complete annual means built from monthly ERA5 sea "
            "surface temperature."
        ),
    )
    return trend, yearly_mean["year"]


def plot_annual_mean_trend(
    input_file: Path = INPUT_FILE,
    figure_file: Path = ANNUAL_FIGURE_FILE,
) -> xr.DataArray:
    trend, years = annual_mean_trend(input_file)
    trend = trend * 10
    figure_path = Path(figure_file)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    valid_values = np.abs(trend.values[np.isfinite(trend.values)])
    if valid_values.size:
        vmax = float(np.nanpercentile(valid_values, 98))
        if not np.isfinite(vmax) or np.isclose(vmax, 0.0):
            vmax = float(np.nanmax(valid_values))
    else:
        vmax = 1.0
    if not np.isfinite(vmax) or np.isclose(vmax, 0.0):
        vmax = 1.0

    levels = np.linspace(-1.6, 1.6, 21)

    fig, ax = plt.subplots(
        figsize=(11, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=CENTRAL_LONGITUDE)},
    )
    lon, lat, data = wrap_global_field(trend)
    contour = ax.contourf(
        lon,
        lat,
        data,
        levels=levels,
        cmap="coolwarm",
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(contour, ax=ax, label="K decade-1", orientation="horizontal", pad=0.04)
    ax.coastlines(linewidth=0.8)
    add_latitude_lines(ax)
    ax.set_global()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Annual Mean skin Trend ({int(years.min())}-{int(years.max())})")
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    return trend


def plot_seasonal_mean_trend(
    season: str,
    output_file: Path = OUTPUT_FILE,
    figure_file: Path | None = None,
) -> xr.DataArray:
    if season not in SEASONS:
        raise ValueError(f"season must be one of {SEASONS}, got {season!r}")

    trend_ds = (
        seasonal_trend(output_file=output_file)
        if not Path(output_file).exists()
        else xr.open_dataset(output_file)
    )
    trend = trend_ds["skin_trend"].sel(season=season) * 10.0

    if figure_file is None:
        figure_file = SCRIPT_DIR / "figures" / f"skin_trend_{season}.png"

    figure_path = Path(figure_file)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    levels = np.linspace(-1.6, 1.6, 21)
    start_year = int(trend_ds["start_year"].sel(season=season))
    end_year = int(trend_ds["end_year"].sel(season=season))

    fig, ax = plt.subplots(
        figsize=(11, 5.5),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.Robinson(central_longitude=CENTRAL_LONGITUDE)},
    )
    lon, lat, data = wrap_global_field(trend)
    contour = ax.contourf(
        lon,
        lat,
        data,
        levels=levels,
        cmap="coolwarm",
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(contour, ax=ax, label="K decade-1", orientation="horizontal", pad=0.04)
    ax.coastlines(linewidth=0.8)
    add_latitude_lines(ax)
    ax.set_global()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{season} Mean skin Trend ({start_year}-{end_year})")
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    return trend


def seasonal_trend(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> xr.Dataset:
    skin = load_monthly_skin(input_file)

    trend_maps = []
    sample_counts = []
    start_years = []
    end_years = []

    for season in SEASONS:
        seasonal_yearly_mean = seasonal_means(skin, season)
        trend_map = fit_trend_map(seasonal_yearly_mean)["slope"].expand_dims(
            season=[season]
        )
        trend_maps.append(trend_map)
        sample_counts.append(int(seasonal_yearly_mean.sizes["year"]))
        start_years.append(int(seasonal_yearly_mean["year"].min()))
        end_years.append(int(seasonal_yearly_mean["year"].max()))

    trend = xr.concat(trend_maps, dim="season").astype(np.float32)
    trend.name = "skin_trend"
    trend.attrs.update(
        long_name="Sea surface temperature seasonal trend",
        units="K yr-1",
        description=(
            "Linear trend from complete seasonal means built from monthly ERA5 sea "
            "surface temperature. Each year contributes four seasonal means; DJF is "
            "assigned to the year of January-February."
        ),
    )

    out = xr.Dataset(
        data_vars={
            "skin_trend": trend,
            "sample_count": xr.DataArray(
                np.asarray(sample_counts, dtype=np.int16),
                coords={"season": list(SEASONS)},
                dims=("season",),
                attrs={
                    "long_name": "Number of complete seasonal means used in each trend fit",
                    "units": "1",
                },
            ),
            "start_year": xr.DataArray(
                np.asarray(start_years, dtype=np.int16),
                coords={"season": list(SEASONS)},
                dims=("season",),
                attrs={
                    "long_name": "First year used in each seasonal trend",
                },
            ),
            "end_year": xr.DataArray(
                np.asarray(end_years, dtype=np.int16),
                coords={"season": list(SEASONS)},
                dims=("season",),
                attrs={
                    "long_name": "Last year used in each seasonal trend",
                },
            ),
        },
        coords={"season": list(SEASONS), "lat": trend["lat"], "lon": trend["lon"]},
        attrs={
            "title": "Seasonal sea surface temperature trend maps",
            "source_file": str(input_file),
            "history": "Created by basic_trend/sea_surface_temperature",
        },
    )

    encoding = {
        "skin_trend": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "sample_count": {"dtype": "int16"},
        "start_year": {"dtype": "int16"},
        "end_year": {"dtype": "int16"},
    }
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(output_file, encoding=encoding)
    return out


if __name__ == "__main__":
    seasonal_trend()
    plot_annual_mean_trend()
    plot_seasonal_mean_trend("DJF", figure_file=DJF_FIGURE_FILE)
