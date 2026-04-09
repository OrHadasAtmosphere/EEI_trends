import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

PLOT_VAR = "net"

# color per regime
colors = {
    "nh_storms": "grey",
    "nh_cryosphere": "cyan",
    "subsidence_land": "goldenrod",
    "subsidence_ocean": "blue",
    "tropical_ascent": "red",
    "sh_storms": "grey",
    "sh_cryosphere": "cyan",
    "residual": "magenta",
}

# hatching per regime
hatches = {
    "nh_storms": "//",
    "sh_storms": "\\",
    "nh_cryosphere": "\\\\",
    "sh_cryosphere": "//",
    "subsidence_land": "//",
    "subsidence_ocean": "xx",
    "tropical_ascent": "\\",
    "residual": "..",
}

ds = xr.open_dataset("pp/regime_mean_trends.nc")
regimes = ds.regime.values

df = ds[PLOT_VAR].to_pandas()
area_ann = ds["area_fraction"].sel(season="ANN").to_pandas()

# --- reorder: ANN first if present ---
if "ANN" in df.index:
    order = ["ANN"] + [s for s in df.index if s != "ANN"]
    df = df.loc[order]

df = df[regimes]

fig, ax = plt.subplots(figsize=(8,4))

bottom_pos = np.zeros(len(df))
bottom_neg = np.zeros(len(df))

spacing = 1.4
x = np.arange(len(df)) * spacing

label_offset = 0.45
threshold = 0.02 * np.max(np.abs(df.values))

for region in df.columns:
    values = df[region].values

    pos = np.clip(values, 0, None)
    neg = np.clip(values, None, 0)

    bars_pos = ax.bar(
        x, pos, bottom=bottom_pos,
        color=colors.get(region, "black"),
        edgecolor="k",
        alpha=0.8,
        label=region,
        width=0.8,
    )

    bars_neg = ax.bar(
        x, neg, bottom=bottom_neg,
        color=colors.get(region, "black"),
        edgecolor="k",
        alpha=0.8,
        width=0.8,
    )

    for i, v in enumerate(values):
        hatch = hatches.get(region, "")
        bars_pos[i].set_hatch(hatch)
        bars_neg[i].set_hatch(hatch)

        if abs(v) > threshold:
            if v >= 0:
                y = bottom_pos[i] + v / 2
            else:
                y = bottom_neg[i] + v / 2

            ax.text(
                x[i] + label_offset,
                y,
                f"{v:.2f}",
                va="center",
                ha="left",
                fontsize=7,
            )

    bottom_pos += pos
    bottom_neg += neg


# --- totals per bar ---
totals = df.sum(axis=1).values
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
    f"{r.replace("_"," ")} ({area_ann[r]*100:.0f}%)"
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
ax.set_xticklabels(df.index)
ax.tick_params(axis="x", which="both", bottom=False, top=False, length=0)

# styling
ax.spines["bottom"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.axvline(0.8, color="k", linewidth=1, ls=":")
ax.axhline(0, color="k", linewidth=0.8)

plt.ylabel("Net EEI trend / W m$^{-2}$ dec$^{-1}$")

plt.savefig(f"figures/trend_barchart_{PLOT_VAR}.png",
            dpi=300, bbox_inches='tight')
