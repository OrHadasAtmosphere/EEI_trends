import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

PLOT_SEASON = "ANN"
AREA_WEIGHT = False

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
if not AREA_WEIGHT:
    ds /= ds.area_fraction

x = np.arange(len(regimes))
width = 0.35  # bar width
x = x - width/2
alpha_light = 0.5

fig, ax = plt.subplots(figsize=(10,4))
for i,sky in enumerate(["clr","cre"]):
    x += width*i
    bottom_pos = np.zeros(len(regimes))
    bottom_neg = np.zeros(len(regimes))

    for comp in ["sw", "lw"]:
        key = comp+"_"+sky
        values = ds[key+"_slope_mean"].values
        ci = ds[key+"_slope_ci"].values
        sig = np.abs(values) > ci

        pos = np.clip(values, 0, None)
        neg = np.clip(values, None, 0)

        for i,xi in enumerate(x):
            ax.bar(x[i] - width/2, pos[i], width,
                bottom=bottom_pos[i],
                color=colors[key],
                alpha=alpha_light + sig[i]*(1-alpha_light),
                label=key,
            )

            ax.bar(x[i] - width/2, neg[i], width,
                bottom=bottom_neg[i],
                color=colors[key],
                alpha=alpha_light + sig[i]*(1-alpha_light),
            )

        bottom_pos += pos
        bottom_neg += neg
    
    net = ds["net_"+sky+"_slope_mean"].values
    offset = 0.2
    if AREA_WEIGHT:
        offset /= 10
    for i in range(len(regimes)):
        ax.text(x[i] - width/2, bottom_pos[i] + offset, f"{net[i]:.2f}",
            ha="center", va="top", fontsize=8)


# --- formatting ---
ax.set_xticks(x-width)
rlabels = [r.replace("_"," ") for r in regimes]
ax.set_xticklabels(rlabels, rotation=45, ha="right")

if AREA_WEIGHT:
    ax.set_ylabel("EEI trend area-weighted / W m$^{-2}$ dec$^{-1}$")
else:
    ax.set_ylabel("EEI trend / W m$^{-2}$ dec$^{-1}$")
ax.axhline(0, color="k", linewidth=0.8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# legend (clean duplicates)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), frameon=False, ncol=2, loc=3)

if AREA_WEIGHT:
    plt.savefig(f"figures/regime_trend_barchart_{PLOT_SEASON}_areaweight.png",
            dpi=300, bbox_inches='tight')
else:
    plt.savefig(f"figures/regime_trend_barchart_{PLOT_SEASON}.png",
            dpi=300, bbox_inches='tight')