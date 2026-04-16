import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

PLOT_SEASON = "ANN"

# --- color scheme (physically intuitive) ---
colors = {
    "sw_clr": "#E69F00",  # warm orange (SW clear)
    "sw_cre": "#D55E00",  # darker orange/red (SW CRE)
    "lw_clr": "#56B4E9",  # light blue (LW clear)
    "lw_cre": "#0072B2",  # dark blue (LW CRE)
}


ds = xr.open_dataset("pp/regime_mean_trends.nc").sel(season=PLOT_SEASON)
ds = ds.sel(regime=[
    "sh_cryosphere", "sh_storms", "subsidence_land",
    "subsidence_ocean", "tropical_ascent",
    "nh_storms", "nh_cryosphere", "residual"
])
regimes = ds.regime.values

x = np.arange(len(regimes))
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(10,4))

x = x - width/2

for i,sky in enumerate(["clr","cre"]):
    x += width*i
    bottom_pos = np.zeros(len(regimes))
    bottom_neg = np.zeros(len(regimes))

    for comp in ["sw", "lw"]:
        key = comp+"_"+sky
        values = ds[key].values
        pos = np.clip(values, 0, None)
        neg = np.clip(values, None, 0)

        ax.bar(x - width/2, pos, width,
           bottom=bottom_pos,
           color=colors[key],
           label=key)

        ax.bar(x - width/2, neg, width,
           bottom=bottom_neg,
           color=colors[key],)

        bottom_pos += pos
        bottom_neg += neg
    
    net = ds["sw_"+sky].values + ds["lw_"+sky].values
    for i in range(len(regimes)):
        ax.text(x[i] - width/2, bottom_pos[i] + 0.02, f"{net[i]:.2f}",
            ha="center", va="top", fontsize=8)


# --- formatting ---
ax.set_xticks(x-width)
rlabels = [r.replace("_"," ") for r in regimes]
ax.set_xticklabels(rlabels, rotation=45, ha="right")

ax.set_ylabel("EEI trend / W m$^{-2}$ dec$^{-1}$")
ax.axhline(0, color="k", linewidth=0.8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# legend (clean duplicates)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), frameon=False, ncol=2, loc=3)

plt.savefig(f"figures/regime_trend_barchart_{PLOT_SEASON}.png",
            dpi=300, bbox_inches='tight')