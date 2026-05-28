import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Patch
from utils.plotting import colors, hatches, hatches_legend
from utils.plotting import central_lon, plot_coasts_grid, edge_band

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
    light_alpha = 0.3

    fig, ax = plt.subplots(figsize=(8,4))

    map_ax = fig.add_axes(
        [0.93, 0.06, 0.25, 0.25],  # [left, bottom, width, height]
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
            bars_pos = ax.bar(x[i], pos[i], width,
                bottom=bottom_pos[i],
                color=colors[r],
                alpha=light_alpha + sig[i]*(1-light_alpha),
                edgecolor="k",
                label=r if i==0 else None,
            )

            bars_neg = ax.bar(x[i], neg[i], width,
                bottom=bottom_neg[i],
                color=colors[r],
                alpha=light_alpha + sig[i]*(1-light_alpha),
                edgecolor="k",
            )    

            hatch = hatches.get(r, "")
            bars_pos[0].set_hatch(hatch)
            bars_neg[0].set_hatch(hatch)

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
                hatch=hatches_legend.get(r, ""),
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
        handleheight=1.3,
        title="Regime (area)",
        alignment="left",
        bbox_to_anchor=(1.02, 1.02),
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

    if PLOT_VAR == "net":
        plt.ylabel(f"{PLOT_VAR[0].upper()+PLOT_VAR[1:]} EEI trend / W m$^{{-2}}$ dec$^{{-1}}$")
    else:
        plt.ylabel(f"{PLOT_VAR.upper()} EEI trend / W m$^{{-2}}$ dec$^{{-1}}$")

    # print numbers
    if PLOT_VAR == "net":
        print(seasons)
        for r in regimes:
            print(r)
            values = ds.sel(regime=r)[PLOT_VAR+"_slope_mean"].values
            ci = ds.sel(regime=r)[PLOT_VAR+"_slope_ci"].values
            print([f"{v:.2f} ± {c:.2f}" for v, c in zip(values, ci)])

    # add map inset
    ds = xr.open_dataset("pp/regime_masks.nc")
    ds = ds.drop_vars("area")
    regimes = [
        "nh_cryosphere",
        "sh_cryosphere",
        "nh_storms",
        "sh_storms",
        "tropical_ascent",
        "subsidence_land",
        "subsidence_ocean",
        "residual",
    ]
    stacked = xr.concat(
        [ds[r] for r in regimes],
        dim=xr.DataArray(regimes, dims="regime", name="regime")
    )
    counts = stacked.sum(dim="season")
    winner_index = counts.argmax(dim="regime")
    annual_ds = xr.Dataset(
        {
            regime: (winner_index == i).astype(int)
            for i, regime in enumerate(regimes)
        }
    )
    plot_coasts_grid(map_ax)

    for r in regimes:
        data = annual_ds[r].astype(int)
        band = edge_band(data, n=2)

        # 1. Colored boundary band
        map_ax.contourf(
            data.lon, data.lat, band,
            levels=[0.5, 1],
            colors=[colors[r]],
            transform=ccrs.PlateCarree(),
        )

        # 2. Filled contours
        map_ax.contourf(
            data.lon, data.lat, data,
            levels=[0.5, 1],
            colors=[colors[r]],
            transform=ccrs.PlateCarree(),
            alpha=light_alpha,
        )

        # 3. Hatches
        map_ax.contourf(
            data.lon, data.lat, data,
            levels=[0.5, 1],
            colors='none',
            hatches=[hatches_legend[r]],
            transform=ccrs.PlateCarree(),
        )
        # trick to color the hatches
        for collection in map_ax.collections[-1:]:  # last contourf
            collection.set_edgecolor(colors[r])
            collection.set_linewidth(0.0)  # remove polygon edges

    plt.savefig(f"figures/{out_subdir}trend_barchart_{PLOT_VAR}{extra_fname}.png",
                dpi=300, bbox_inches='tight')

for var in ['net', 'sw']:
    do_barchart(PLOT_VAR=var)
