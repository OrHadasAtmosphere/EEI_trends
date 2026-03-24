import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from map_plot_utils import wrap_global_field

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = BASE_DIR / "figures"

SEASONS = ("DJF", "MAM", "JJA", "SON")
PERIODS = ("annual", *SEASONS)
SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
STORM_TRACK_NH_FACTOR = 0.3
STORM_TRACK_SH_FACTOR = 0.4
OMEGA_POS_FACTOR = 0.05
OMEGA_NEG_FACTOR = 0.05
ICE_FACTOR = 0.1
OMEGA_LATITUDE_LIMIT = 40.0
PLOT_VARIABLES = {
    "all_sky_net_trend": "All-sky net trend",
    "all_sky_shortwave_trend": "All-sky shortwave trend",
    "all_sky_longwave_trend": "All-sky longwave trend",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot seasonal CERES trend maps and report regional contributions."
    )
    parser.add_argument(
        "--all-trends",
        action="store_true",
        help="Plot net, shortwave, and longwave trends. Default is net trend only.",
    )
    return parser.parse_args()


def limits(slp: xr.DataArray, omega: xr.DataArray) -> tuple[float, float, float, float]:
    nh_limit = float(slp.where(slp["lat"] > 0).max(skipna=True)) * STORM_TRACK_NH_FACTOR
    sh_limit = float(slp.where(slp["lat"] < 0).max(skipna=True)) * STORM_TRACK_SH_FACTOR

    omega_pos = omega.where(omega > 0)
    omega_neg = omega.where(omega < 0)
    pos_limit = float(omega_pos.max(skipna=True)) * OMEGA_POS_FACTOR
    neg_limit = float(omega_neg.min(skipna=True)) * OMEGA_NEG_FACTOR
    return nh_limit, sh_limit, pos_limit, neg_limit


def regrid_to_target(
    da: xr.DataArray, target_lat: xr.DataArray, target_lon: xr.DataArray
) -> xr.DataArray:
    return da.sortby("lat").interp(
        lat=target_lat,
        lon=target_lon,
        kwargs={"fill_value": "extrapolate"},
    )


def load_contours() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    omega_ds = xr.open_dataset(OUTPUT_DIR / "W_mean.nc")
    omega_seasonal = (
        omega_ds["w"].sel(season=list(SEASONS)).rename(season="period").sortby("lat")
    )
    omega_annual = omega_seasonal.mean("period", skipna=True).expand_dims(
        period=["annual"]
    )
    omega = xr.concat([omega_annual, omega_seasonal], dim="period").assign_coords(
        period=list(PERIODS)
    )

    slp_monthly = xr.open_dataset(OUTPUT_DIR / "climatology_ERA5.nc")["SLP_var"].sortby(
        "lat"
    )
    slp_fields = [slp_monthly.mean("month", skipna=True).expand_dims(period=["annual"])]
    for season in SEASONS:
        seasonal_mean = slp_monthly.sel(month=SEASON_MONTHS[season]).mean(
            "month", skipna=True
        )
        slp_fields.append(seasonal_mean.expand_dims(period=[season]))
    slp = xr.concat(slp_fields, dim="period").assign_coords(period=list(PERIODS))

    sin_lat = xr.DataArray(
        np.sin(np.deg2rad(slp["lat"].data)),
        coords={"lat": slp["lat"]},
        dims=("lat",),
    )
    sin_lat = xr.where(np.abs(slp["lat"]) < 20, 1.0, sin_lat)

    ice_ds = xr.open_dataset(OUTPUT_DIR / "ice_mean.nc")
    ice_seasonal = (
        ice_ds["siconc"].sel(season=list(SEASONS)).rename(season="period").sortby("lat")
    )
    ice_annual = ice_seasonal.mean("period", skipna=True).expand_dims(period=["annual"])
    ice = xr.concat([ice_annual, ice_seasonal], dim="period").assign_coords(
        period=list(PERIODS)
    )

    return omega, slp / (sin_lat**2), ice


def build_region_masks(
    slp: xr.DataArray, omega: xr.DataArray, ice: xr.DataArray
) -> dict[str, xr.DataArray]:
    slp_nh_limit, slp_sh_limit, pos_limit, neg_limit = limits(slp, omega)
    lat2d, _ = xr.broadcast(slp["lat"], slp["lon"])
    ice_mask = ice > ICE_FACTOR
    nh_ice = (lat2d > 0) & ice_mask
    sh_ice = (lat2d < 0) & ice_mask

    nh_storm = (lat2d > 0) & (~ice_mask) & (slp >= slp_nh_limit)
    sh_storm = (lat2d < 0) & (~ice_mask) & (slp >= slp_sh_limit)
    storm_track = nh_storm | sh_storm
    equatorward = np.abs(lat2d) < OMEGA_LATITUDE_LIMIT

    nh_positive_omega = (
        (lat2d > 0) & equatorward & (~storm_track) & (~ice_mask) & (omega >= pos_limit)
    )
    sh_positive_omega = (
        (lat2d < 0) & equatorward & (~storm_track) & (~ice_mask) & (omega >= pos_limit)
    )
    neg_omega = equatorward & (~storm_track) & (~ice_mask) & (omega <= neg_limit)
    midlatitude_non_storm = (~storm_track) & (~equatorward) & (~ice_mask)
    tropical_remainder = (
        equatorward
        & (~storm_track)
        & (~ice_mask)
        & (~nh_positive_omega)
        & (~sh_positive_omega)
        & (~neg_omega)
    )

    return {
        "NH ice": nh_ice,
        "SH ice": sh_ice,
        "NH storm track": nh_storm,
        "SH storm track": sh_storm,
        "NH positive omega": nh_positive_omega,
        "SH positive omega": sh_positive_omega,
        "negative omega": neg_omega,
        "midlatitude non-storm": midlatitude_non_storm,
        "tropical remainder": tropical_remainder,
    }


def weighted_region_statistics(
    data: xr.DataArray,
    mask: xr.DataArray,
    lat_weights: xr.DataArray,
) -> tuple[float, float, float, float]:
    valid_mask = mask & data.notnull()
    weighted_mask = lat_weights.broadcast_like(data).where(valid_mask, 0.0)
    weight_sum = float(weighted_mask.sum(skipna=True))
    if weight_sum == 0.0:
        return np.nan, np.nan, np.nan, 0.0

    weighted_sum = float((data.where(valid_mask, 0.0) * weighted_mask).sum(skipna=True))
    weighted_sq_sum = float(
        ((data.where(valid_mask, 0.0) ** 2) * weighted_mask).sum(skipna=True)
    )

    mean = weighted_sum / weight_sum
    variance = max(weighted_sq_sum / weight_sum - mean**2, 0.0)
    return weighted_sum, mean, float(np.sqrt(variance)), weight_sum


def percent_variability(variability: float, mean: float) -> float:
    if not np.isfinite(variability) or not np.isfinite(mean) or np.isclose(mean, 0.0):
        return np.nan
    return 100.0 * variability / abs(mean)


def smart_contribution(
    trend: xr.DataArray,
    slp_fields: xr.DataArray,
    omega_fields: xr.DataArray,
    ice_fields: xr.DataArray,
) -> xr.Dataset:
    seasonal_trend = trend.sel(period=list(SEASONS)) * 4.0
    target_lat = seasonal_trend["lat"]
    target_lon = seasonal_trend["lon"]
    seasonal_slp = regrid_to_target(
        slp_fields.sel(period=list(SEASONS)), target_lat, target_lon
    )
    seasonal_omega = regrid_to_target(
        omega_fields.sel(period=list(SEASONS)), target_lat, target_lon
    )
    seasonal_ice = regrid_to_target(
        ice_fields.sel(period=list(SEASONS)), target_lat, target_lon
    )

    weights = xr.DataArray(
        np.cos(np.deg2rad(target_lat.data)),
        coords={"lat": target_lat},
        dims=("lat",),
    )
    region_names = list(
        build_region_masks(
            seasonal_slp.sel(period=SEASONS[0]),
            seasonal_omega.sel(period=SEASONS[0]),
            seasonal_ice.sel(period=SEASONS[0]),
        ).keys()
    )

    regional_sums = np.zeros((len(SEASONS), len(region_names)), dtype=np.float64)
    regional_means = np.zeros((len(SEASONS), len(region_names)), dtype=np.float64)
    regional_variability = np.zeros((len(SEASONS), len(region_names)), dtype=np.float64)
    regional_weight_sums = np.zeros((len(SEASONS), len(region_names)), dtype=np.float64)
    regional_weighted_sq_sums = np.zeros(
        (len(SEASONS), len(region_names)), dtype=np.float64
    )
    global_sums = np.zeros(len(SEASONS), dtype=np.float64)

    for season_index, season in enumerate(SEASONS):
        season_trend = seasonal_trend.sel(period=season)
        season_global_sum = float((season_trend * weights).sum(skipna=True))
        global_sums[season_index] = season_global_sum

        masks = build_region_masks(
            seasonal_slp.sel(period=season),
            seasonal_omega.sel(period=season),
            seasonal_ice.sel(period=season),
        )
        for region_index, region_name in enumerate(region_names):
            region_sum, region_mean, region_std, region_weight_sum = (
                weighted_region_statistics(
                    season_trend,
                    masks[region_name],
                    weights,
                )
            )
            regional_sums[season_index, region_index] = region_sum
            regional_means[season_index, region_index] = region_mean
            regional_variability[season_index, region_index] = region_std
            regional_weight_sums[season_index, region_index] = region_weight_sum
            regional_weighted_sq_sums[season_index, region_index] = (
                region_weight_sum * (region_std**2 + region_mean**2)
            )

    overall_region_sums = regional_sums.sum(axis=0)
    overall_global_sum = global_sums.sum()
    contributions = regional_sums / overall_global_sum
    overall_contributions = overall_region_sums / overall_global_sum
    overall_weight_sums = regional_weight_sums.sum(axis=0)
    overall_means = overall_region_sums / overall_weight_sums
    overall_variability = np.sqrt(
        np.maximum(
            regional_weighted_sq_sums.sum(axis=0) / overall_weight_sums
            - overall_means**2,
            0.0,
        )
    )
    regional_variability_percent = np.array(
        [
            [
                percent_variability(
                    regional_variability[season_index, region_index],
                    regional_means[season_index, region_index],
                )
                for region_index in range(len(region_names))
            ]
            for season_index in range(len(SEASONS))
        ],
        dtype=np.float64,
    )
    overall_variability_percent = np.array(
        [
            percent_variability(
                overall_variability[region_index], overall_means[region_index]
            )
            for region_index in range(len(region_names))
        ],
        dtype=np.float64,
    )

    return xr.Dataset(
        data_vars={
            "regional_sum": (("season", "region"), regional_sums),
            "regional_mean": (("season", "region"), regional_means),
            "regional_variability": (("season", "region"), regional_variability),
            "regional_variability_percent": (
                ("season", "region"),
                regional_variability_percent,
            ),
            "contribution": (("season", "region"), contributions),
            "global_sum": ("season", global_sums),
            "overall_regional_sum": ("region", overall_region_sums),
            "overall_regional_mean": ("region", overall_means),
            "overall_regional_variability": ("region", overall_variability),
            "overall_regional_variability_percent": (
                "region",
                overall_variability_percent,
            ),
            "overall_contribution": ("region", overall_contributions),
        },
        coords={"season": list(SEASONS), "region": region_names},
    )


def print_contribution_summary(summary: xr.Dataset) -> None:
    for season in SEASONS:
        print(f"\nSeason: {season}")
        for region in summary["region"].values:
            contribution = float(
                summary["contribution"].sel(season=season, region=region)
            )
            variability_percent = float(
                summary["regional_variability_percent"].sel(
                    season=season, region=region
                )
            )
            print(
                f"  {region}: contribution={contribution:.3%}, variability={variability_percent:.3f}%"
            )

    print("\nOverall contribution from summed seasonal totals")
    for region in summary["region"].values:
        contribution = float(summary["overall_contribution"].sel(region=region))
        variability_percent = float(
            summary["overall_regional_variability_percent"].sel(region=region)
        )
        print(
            f"  {region}: contribution={contribution:.3%}, variability={variability_percent:.3f}%"
        )


def add_row_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        0.5,
        label,
        transform=ax.transAxes,
        va="center",
        ha="right",
        fontsize=12,
        fontweight="bold",
    )


def plot_map(
    ax: plt.Axes,
    data: xr.DataArray,
    omega: xr.DataArray,
    slp: xr.DataArray,
    ice: xr.DataArray,
    cmap: str,
    levels: np.ndarray,
    title: str,
) -> None:
    data_lon, data_lat, data_values = wrap_global_field(data)
    ax.contourf(
        data_lon,
        data_lat,
        data_values,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        levels=levels,
    )
    ax.coastlines()
    if title:
        ax.set_title(title)
    ice_lon, ice_lat, ice_values = wrap_global_field(ice)
    ax.contour(
        ice_lon,
        ice_lat,
        ice_values,
        levels=[ICE_FACTOR, float(ice.max(skipna=True)) + 1e-6],
        colors=["green"],
        transform=ccrs.PlateCarree(),
    )

    slp_nh_limit, slp_sh_limit, pos_limit, neg_limit = limits(slp, omega)
    slp_nh_lon, slp_nh_lat, slp_nh_values = wrap_global_field(slp.where(slp["lat"] > 0))
    ax.contour(
        slp_nh_lon,
        slp_nh_lat,
        slp_nh_values,
        levels=[slp_nh_limit],
        colors="black",
        linewidths=2,
        transform=ccrs.PlateCarree(),
    )
    slp_sh_lon, slp_sh_lat, slp_sh_values = wrap_global_field(slp.where(slp["lat"] < 0))
    ax.contour(
        slp_sh_lon,
        slp_sh_lat,
        slp_sh_values,
        levels=[slp_sh_limit],
        colors="black",
        linewidths=2,
        transform=ccrs.PlateCarree(),
    )
    omega_lon, omega_lat, omega_values = wrap_global_field(omega.where(np.abs(omega["lat"]) < 45))
    ax.contour(
        omega_lon,
        omega_lat,
        omega_values,
        levels=[neg_limit, pos_limit],
        colors=["red", "blue"],
        linewidths=2,
        transform=ccrs.PlateCarree(),
    )


def main() -> None:
    args = parse_args()
    ds = xr.open_dataset(OUTPUT_DIR / "ceres_ebaf_toa_trend_maps.nc")
    omega, slp, ice = load_contours()

    summary = smart_contribution(ds["all_sky_net_trend"], slp, omega, ice)
    print_contribution_summary(summary)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    vars_to_plot = (
        ["all_sky_net_trend"]
        if not args.all_trends
        else [
            "all_sky_net_trend",
            "all_sky_shortwave_trend",
            "all_sky_longwave_trend",
        ]
    )
    figure_width = 8 if len(vars_to_plot) == 1 else 15
    fig, axs = plt.subplots(
        5,
        len(vars_to_plot),
        squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=100)},
        figsize=(figure_width, 20),
    )
    for row_index, period in enumerate(PERIODS):
        for column_index, var_name in enumerate(vars_to_plot):
            plot_map(
                axs[row_index, column_index],
                ds[var_name].sel(period=period),
                omega.sel(period=period),
                slp.sel(period=period),
                ice.sel(period=period),
                cmap="RdBu_r",
                levels=np.linspace(-1, 1, 21) * 0.7,
                title=PLOT_VARIABLES[var_name] if row_index == 0 else "",
            )
        add_row_label(axs[row_index, 0], period)

    fig.tight_layout(rect=(0.08, 0.0, 1.0, 1.0))
    plt.savefig(FIGURES_DIR / "trends.png", dpi=400)


if __name__ == "__main__":
    main()
