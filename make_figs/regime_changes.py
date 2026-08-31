"""Plot seasonal regime changes and their annual net-EEI trend contributions."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from obs_io.process_masks import (
    POLAR_LAT,
    TROPICAL_LAT,
    LAND_THRESH,
    OMEGA_THRESH,
    SIC_THRESH,
    STORM_PROXIMITY_THRESH,
    smooth,
)
from utils.plotting import central_lon, plot_coasts_grid, colors


INPUT_EARLY_CLIMATOLOGY = Path("pp/era5_clim.nc")
INPUT_LATE_CLIMATOLOGY = Path("pp/diff_yrs/era5_clim_2015.nc")
INPUT_EEI_TRENDS = Path("pp/ceres_trends.nc")
OUTPUT_DATA = Path("pp/regime_change_eei_contributions.nc")
OUTPUT_FIGURE = Path("figures/regime_changes_eei.png")

SEASONS = ["MAM", "JJA", "SON", "DJF"]
SEASON_DAYS = xr.DataArray(
    [90.65, 92, 92, 91],
    coords={"season": ["DJF", "MAM", "JJA", "SON"]},
    dims="season"
)
REGIMES = [
    "storms",
    "tropical_ascent",
    "subsidence_land",
    "subsidence_ocean",
    "cryosphere",
]
REGIME_LABELS = {
    "storms": "Storms",
    "tropical_ascent": "Trop. ascent",
    "subsidence_land": "Sub. land",
    "subsidence_ocean": "Sub. ocean",
    "cryosphere": "Cryosphere",
}
REGIME_CODES = {name: index + 1 for index, name in enumerate(REGIMES)}
STATUS_LABELS = {
    "unchanged": "unchanged",
    "added_from_blank": "added from resid.",
    "removed_to_blank": "removed to resid.",
}


def _blend(color, target, fraction):
    rgb = np.asarray(mcolors.to_rgb(color))
    target_rgb = np.asarray(mcolors.to_rgb(target))
    return mcolors.to_hex((1.0 - fraction) * rgb + fraction * target_rgb)

def _status_color(regime, status):
    color_key = {
        "storms": "nh_storms",
        "cryosphere": "nh_cryosphere",
        "subsidence_land": "subsidence_land",
        "subsidence_ocean": "subsidence_ocean",
        "tropical_ascent": "tropical_ascent",
    }[regime]

    base = colors[color_key]

    if status == "added_from_blank":
        return _blend(base, "black", 0.2)
    if status == "removed_to_blank":
        return _blend(base, "white", 0.8)
    return base


def _cryosphere_mask(lsm, siconc):
    nh = (
        ((lsm.lat >= POLAR_LAT) & (siconc > SIC_THRESH))
        | ((lsm.lat >= 75) & (lsm > LAND_THRESH))
        | (
            (lsm.lat >= POLAR_LAT)
            & (lsm > LAND_THRESH)
            & (lsm.lon > 300)
            & (lsm.lon < 350)
        )
    )
    sh = (
        ((lsm.lat <= -POLAR_LAT) & (siconc > SIC_THRESH))
        | ((lsm.lat <= -65) & (lsm > LAND_THRESH))
    )
    return nh | sh


def _regime_state(lsm, cryosphere, storm, ascent, descent):
    state = xr.zeros_like(lsm, dtype=np.int8)
    state = xr.where(
        cryosphere,
        REGIME_CODES["cryosphere"],
        state,
    )
    available = state == 0
    state = xr.where(
        available & storm,
        REGIME_CODES["storms"],
        state,
    )
    within_tropics = (lsm.lat >= -TROPICAL_LAT) & (lsm.lat <= TROPICAL_LAT)
    available = state == 0
    state = xr.where(
        available & within_tropics & ascent,
        REGIME_CODES["tropical_ascent"],
        state,
    )
    available = state == 0
    state = xr.where(
        available & within_tropics & descent & (lsm >= LAND_THRESH),
        REGIME_CODES["subsidence_land"],
        state,
    )
    available = state == 0
    state = xr.where(
        available & within_tropics & descent & (lsm < LAND_THRESH),
        REGIME_CODES["subsidence_ocean"],
        state,
    )
    return state.astype(np.int8)

def calculate_period_states(clim_early, clim_late):
    lsm = clim_early.lsm

    early_siconc = clim_early.siconc
    early_cryosphere = _cryosphere_mask(lsm, early_siconc)
    early_omega = smooth(clim_early.omega500)
    early_storm = smooth(clim_early.monthly_storm_day_fraction)
    early = _regime_state(
        lsm,
        early_cryosphere,
        early_storm > STORM_PROXIMITY_THRESH,
        early_omega <= -OMEGA_THRESH,
        early_omega > OMEGA_THRESH,
    )

    late_siconc = clim_late.siconc
    late_cryosphere = _cryosphere_mask(lsm, late_siconc)
    late_omega = smooth(clim_late.omega500)
    late_storm = smooth(clim_late.monthly_storm_day_fraction)
    late = _regime_state(
        lsm,
        late_cryosphere,
        late_storm > STORM_PROXIMITY_THRESH,
        late_omega <= -OMEGA_THRESH,
        late_omega > OMEGA_THRESH,
    )
    return early, late


def _transition_names():
    return [
        f"{source}_to_{destination}"
        for source in REGIMES
        for destination in REGIMES
        if source != destination
    ]


def _transition_label(name):
    source, destination = name.split("_to_")
    return f"{REGIME_LABELS[source]} → {REGIME_LABELS[destination]}"


def _category_definitions(early, late):
    definitions = []
    for regime in REGIMES:
        code = REGIME_CODES[regime]
        definitions.extend(
            [
                (
                    f"{regime}_unchanged",
                    (early == code) & (late == code),
                    _status_color(regime, "unchanged"),
                    f"{REGIME_LABELS[regime]}: unchanged",
                ),
                (
                    f"blank_to_{regime}",
                    (early == 0) & (late == code),
                    _status_color(regime, "added_from_blank"),
                    f"{REGIME_LABELS[regime]}: added from blank",
                ),
                (
                    f"{regime}_to_blank",
                    (early == code) & (late == 0),
                    _status_color(regime, "removed_to_blank"),
                    f"{REGIME_LABELS[regime]}: removed to blank",
                ),
            ]
        )

    transition_colors = {}
    palette = plt.get_cmap("tab20").colors
    for index, name in enumerate(_transition_names()):
        source, destination = name.split("_to_")
        transition_colors[name] = mcolors.to_hex(palette[index])
        definitions.append(
            (
                name,
                (early == REGIME_CODES[source])
                & (late == REGIME_CODES[destination]),
                transition_colors[name],
                _transition_label(name),
            )
        )
    return definitions, transition_colors


def _global_contribution(field, mask, area):
    return (field * mask).weighted(area).mean(("lat", "lon"))


def calculate_contributions(early, late, eei_trend, area):
    components = ["unchanged", "added_from_blank", "removed_to_blank"]
    components.extend(_transition_names())
    contribution = xr.DataArray(
        np.zeros((len(SEASONS), len(REGIMES), len(components))),
        coords={"season": SEASONS, "regime": REGIMES, "component": components},
        dims=("season", "regime", "component"),
        name="net_eei_trend_contribution",
        attrs={"units": "W m-2 decade-1"},
    )
    area_fraction = xr.zeros_like(contribution).rename("area_fraction")
    global_area = area.sum(("lat", "lon"))

    for regime in REGIMES:
        code = REGIME_CODES[regime]
        masks = {
            "unchanged": (early == code) & (late == code),
            "added_from_blank": (early == 0) & (late == code),
            "removed_to_blank": (early == code) & (late == 0),
        }
        for transition in _transition_names():
            source, destination = transition.split("_to_")
            if regime not in (source, destination):
                continue
            masks[transition] = (
                (early == REGIME_CODES[source])
                & (late == REGIME_CODES[destination])
            )
        for component, mask in masks.items():
            contribution.loc[dict(regime=regime, component=component)] = (
                _global_contribution(eei_trend, mask, area)
            )
            area_fraction.loc[dict(regime=regime, component=component)] = (
                (mask * area).sum(("lat", "lon")) / global_area
            )

    annual_contribution = contribution.weighted(SEASON_DAYS).mean("season")
    annual_area_fraction = area_fraction.weighted(SEASON_DAYS).mean("season")
    contribution = xr.concat(
        [
            contribution,
            annual_contribution.expand_dims(season=["ANN"]),
        ],
        dim="season",
    )
    area_fraction = xr.concat(
        [area_fraction, annual_area_fraction.expand_dims(season=["ANN"])],
        dim="season",
    )
    return xr.Dataset(
        {
            "net_eei_trend_contribution": contribution,
            "area_fraction": area_fraction,
        }
    )


def _plot_maps(axes, early, late, definitions):
    colors = ["white"] + [definition[2] for definition in definitions]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(
        np.arange(-0.5, len(colors) + 0.5),
        cmap.N,
    )
    categories = xr.zeros_like(early, dtype=np.int16)
    category_count = xr.zeros_like(early, dtype=np.int8)
    for code, (_, mask, _, _) in enumerate(definitions, start=1):
        categories = xr.where(mask, code, categories)
        category_count = category_count + mask.astype(np.int8)
    expected_coverage = (early != 0) | (late != 0)
    assert bool((category_count <= 1).all()), "regime-change categories overlap"
    assert bool(((category_count > 0) == expected_coverage).all()), (
        "regime-change categories do not cover every classified grid cell"
    )

    for index, season in enumerate(SEASONS):
        axis = axes.flat[index]
        data = categories.sel(season=season)
        axis.pcolormesh(
            data.lon,
            data.lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading="auto",
        )
        plot_coasts_grid(axis)
        axis.set_global()
        axis.set_title(f"{chr(97 + index)}) {season}", loc="left")
    return categories


def _component_color(regime, component, transition_colors):
    if component in STATUS_LABELS:
        return _status_color(regime, component)
    return transition_colors[component]


def _plot_bars(axis, contributions, transition_colors):
    annual = contributions.net_eei_trend_contribution.sel(season="ANN")
    x = np.arange(len(REGIMES))
    positive_bottom = np.zeros(len(REGIMES))
    negative_bottom = np.zeros(len(REGIMES))
    legend = {}

    for regime_index, regime in enumerate(REGIMES):
        components = ["unchanged", "added_from_blank", "removed_to_blank"]
        components.extend(
            transition
            for transition in _transition_names()
            if regime in transition.split("_to_")
        )
        for component in components:
            value = float(annual.sel(regime=regime, component=component))
            color = _component_color(regime, component, transition_colors)
            if component in STATUS_LABELS:
                label = f"{REGIME_LABELS[regime]}: {STATUS_LABELS[component]}"
            else:
                label = _transition_label(component)
            bottom = positive_bottom if value >= 0 else negative_bottom
            axis.bar(
                x[regime_index],
                value,
                width=0.62,
                bottom=bottom[regime_index],
                color=color,
                edgecolor="black",
                linewidth=0.45,
            )
            bottom[regime_index] += value
            legend[label] = color

    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([REGIME_LABELS[name] for name in REGIMES])
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("Annual net EEI trend contribution / W m$^{-2}$ dec$^{-1}$", y=0.98)
    return legend

def _legend_sort_key(item):
    label, _ = item

    for regime in REGIMES:
        regime_label = REGIME_LABELS[regime]
        if label.startswith(f"{regime_label}:"):
            return (REGIMES.index(regime), 0)

        if label.startswith(f"{regime_label} →"):
            return (REGIMES.index(regime), 1)

    return (len(REGIMES), 0)


def make_figure(early, late, contributions, output_path=OUTPUT_FIGURE):
    definitions, transition_colors = _category_definitions(early, late)
    projection = ccrs.Robinson(central_longitude=central_lon)
    figure = plt.figure(figsize=(9, 8))
    grid = figure.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 0.05, 1.1])
    map_axes = np.asarray(
        [
            figure.add_subplot(grid[row, column], projection=projection)
            for row in range(2)
            for column in range(2)
        ]
    ).reshape(2, 2)
    bar_axis = figure.add_subplot(grid[3, :])

    categories = _plot_maps(map_axes, early, late, definitions)
    bar_legend = _plot_bars(bar_axis, contributions, transition_colors)

    bar_handles = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for label, color in sorted(bar_legend.items(), key=_legend_sort_key)
    ]
    figure.legend(
        bar_handles,
        [handle.get_label() for handle in bar_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
        frameon=False,
        fontsize=6,
    )
    
    figure.suptitle(
        "Seasonal regime changes and their contributions to net EEI trends:\n"
        "Late (2015-2025) minus Early (1990-2000) period",
        fontsize=12,
    )
    figure.subplots_adjust(hspace=0.15, wspace=0.03, bottom=0.16, top=0.88)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    return figure


def main():
    clim_early = xr.open_dataset(INPUT_EARLY_CLIMATOLOGY).load()
    clim_late = xr.open_dataset(INPUT_LATE_CLIMATOLOGY).load()
    early, late = calculate_period_states(clim_early, clim_late)

    with xr.open_dataset(INPUT_EEI_TRENDS) as dataset:
        eei_trend = dataset.net.sel(season=SEASONS).load()
    latitude_area = np.cos(np.deg2rad(early.lat))
    area = latitude_area.broadcast_like(early)
    contributions = calculate_contributions(early, late, eei_trend, area)
    contributions.attrs.update(
        {
            "description": "Area-weighted net EEI trend contributions from regime changes",
            "early_period": "1990-2000",
            "late_period": "2015-2025",
        }
    )
    contributions.to_netcdf(OUTPUT_DATA)
    figure = make_figure(early, late, contributions)
    plt.close(figure)
    print(f"wrote {OUTPUT_DATA}")
    print(f"wrote {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
