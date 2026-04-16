import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

PLOT_VAR = "net"

# color per regime
colors = {
    "nh_storms": "grey",
    "nh_cryosphere": "cyan",
    "subsidence_land": "goldenrod",
    "subsidence_ocean": "lime",
    "tropical_ascent": "magenta",
    "sh_storms": "grey",
    "sh_cryosphere": "cyan",
    "residual": "lightpink",
}

# hatching per regime
hatches = {
    "nh_storms": "/",
    "sh_storms": "\\",
    "nh_cryosphere": "/",
    "sh_cryosphere": "\\",
    "subsidence_land": "/",
    "subsidence_ocean": "/",
    "tropical_ascent": "\\",
    "residual": "..",
}

ds = xr.open_dataset("pp/regime_mean_trends.nc")

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

bottom_pos = np.zeros(len(seasons))
bottom_neg = np.zeros(len(seasons))

fig, ax = plt.subplots(figsize=(8,4))

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
            label=r,
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
        if pct > 3:
            if values[i]:
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
        ax.text(x[i], bottom_pos[i] + offset, f"{total:.2f}",
                ha="center", va="bottom", fontsize=8)
    else:
        ax.text(x[i], bottom_neg[i] - offset, f"{total:.2f}",
                ha="center", va="top", fontsize=8)

# legend cleanup
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# build new labels with area fraction
legend_labels = [
    f"{r.replace("_"," ")} ({area_ann.sel(regime=r)*100:.0f}%)"
    for r in by_label.keys()
]

ax.legend(
    by_label.values(),
    legend_labels,
    title="Regime",
    bbox_to_anchor=(1.0, 1.0),
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

plt.ylabel("Net EEI trend / W m$^{-2}$ dec$^{-1}$")

plt.savefig(f"figures/trend_barchart_{PLOT_VAR}.png",
            dpi=300, bbox_inches='tight')
