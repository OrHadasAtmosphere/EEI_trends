import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils.plotting import colors, hatches, hatches_legend, central_lon, plot_coasts_grid, edge_band


def do_barchart(regime_trend_file="pp/regime_mean_trends.nc", PLOT_VAR="net", extra_fname="", out_subdir=""):
    ds = xr.open_dataset(regime_trend_file)

    ds = ds.sel(season=[
        "ANN","MAM","JJA","SON","DJF",
    ])

    area_ann = ds["area_fraction"].sel(season="ANN")

    regimes = ds.regime.values
    seasons = ds.season.values
    x = np.arange(len(seasons))
    width = 0.5
    label_offset = 0.3
    light_alpha = 0.5

    fig, ax = plt.subplots(figsize=(8,4))

    map_ax = fig.add_axes(
        [0.93, 0.05, 0.25, 0.25],  # [left, bottom, width, height]
        projection=ccrs.Robinson(central_longitude=central_lon)
    )

    bottom_pos = np.zeros(len(seasons))
    bottom_neg = np.zeros(len(seasons))

    totals = ds[PLOT_VAR+"_slope_mean"].sum("regime").values
    for r in regimes:
        values = ds.sel(regime=r)[PLOT_VAR+"_slope_mean"].values
        ci = ds.sel(regime=r)[PLOT_VAR+"_slope_ci"].values
        sig = np.abs(values) > ci

        pos = np.clip(values, 0, None)
        neg = np.clip(values, None, 0)

        # need to loop for alpha
        for i,xi in enumerate(x):
            is_sig = sig[i]
            bars_pos = ax.bar(x[i], pos[i], width,
                bottom=bottom_pos[i],
                color=colors[r],
                # alpha=light_alpha + sig[i]*(1-light_alpha),
                alpha=0.6,
                edgecolor="k",
                label=r if i==0 else None,
            )

            bars_neg = ax.bar(x[i], neg[i], width,
                bottom=bottom_neg[i],
                color=colors[r],
                # alpha=light_alpha + sig[i]*(1-light_alpha),
                alpha=0.6,
                edgecolor="k",
            )    

            # hatch = hatches.get(r, "")
            if not is_sig:
                bars_pos[0].set_hatch("oo")
                bars_neg[0].set_hatch("oo")

            pct = values[i]/totals[i]*100
            if np.abs(pct) > 5:
                if values[i] > 0:
                    y = bottom_pos[i] + values[i] / 2
                else:
                    y = bottom_neg[i] + values[i] / 2
                ax.text(
                    x[i]+label_offset, y,
                    f"{pct:.0f}%",
                    va="center",
                    ha="left",
                    fontsize=8,
                    fontweight=400+300*sig[i],
                )

        bottom_pos += pos
        bottom_neg += neg

    # --- totals per bar ---
    offset = 0.02 * np.max(np.abs(totals)) if len(totals) > 0 else 0

    for i, total in enumerate(totals):
        if total >= 0:
            ax.text(x[i] - width/4, bottom_pos[i] + offset, f"{total:.2f}",
                    ha="center", va="bottom", fontsize=8)
        else:
            ax.text(x[i] - width/4, bottom_neg[i] - offset, f"{total:.2f}",
                    ha="center", va="top", fontsize=8)


    # legend clean 
    legend_handles = []
    legend_labels = []

    for r in regimes:
        legend_handles.append(
            Patch(
                facecolor=colors[r],
                edgecolor="k",
                # hatch=hatches_legend.get(r, ""),
                alpha=1.0
            )
        )
        r_formatted = r.replace('_',' ').replace("nh","NH").replace("sh","SH")
        r_formatted = r_formatted[0].upper() + r_formatted[1:]
        legend_labels.append(
            f"{r_formatted} ({area_ann.sel(regime=r)*100:.0f}%)"
        )

    ax.legend(
        legend_handles,
        legend_labels,
        handleheight=1.5,
        title="Regime (area)",
        alignment="left",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False
    )

    # axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.tick_params(axis="x", which="both", bottom=False, top=False, length=0)

    # styling
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.axvline(0.6, color="k", linewidth=1, ls=":")
    ax.axhline(0, color="k", linewidth=0.8)

    plt.ylabel(f"{PLOT_VAR.upper()} EEI trend / W m$^{{-2}}$ dec$^{{-1}}$")

    # -- inset figure of annual mean masks -- #
    s = "ANN"
    ann_mean_regimes = xr.open_dataset("pp/regime_masks_ann.nc", engine='netcdf4')
    plot_coasts_grid(map_ax)

    for r in regimes:
        data = ann_mean_regimes[r].sel(season=s).astype(int)
        band = edge_band(data, n=2)

        # 1. Colored boundary band (replaces contour)
        map_ax.contourf(
            data.lon, data.lat, band,
            levels=[0.5, 1],
            colors=[colors[r]],
            transform=ccrs.PlateCarree(),
        )

        # 2. Hatched regions with colored hatches
        map_ax.contourf(
            data.lon, data.lat, data,
            levels=[0.5, 1],
            colors=[colors[r]],
            transform=ccrs.PlateCarree(),
            alpha=0.35,
        )

    plt.savefig(f"figures/{out_subdir}trend_barchart_{PLOT_VAR}{extra_fname}.png",
                dpi=300, bbox_inches='tight')

for var in ['net', 'sw']:
    do_barchart(PLOT_VAR=var)