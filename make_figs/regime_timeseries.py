import xarray as xr
import matplotlib.pyplot as plt
from utils.plotting import colors

PLOT_VAR = "net"

ds = xr.open_dataset("pp/regime_mean_trends.nc")
fig,axes=plt.subplots(2,4,figsize=(12,4),sharex=True,constrained_layout=True)
for i,r in enumerate(ds.regime):
    name = str(r.values)
    ax=axes.flatten()[i]
    da = ds.sel(season="ANN", regime=r)
    da[PLOT_VAR].plot(ax=ax, color=colors[str(r.values)])
    da[PLOT_VAR+"_fit"].plot(ax=ax, color=colors[str(r.values)], label=f"{da[PLOT_VAR+"_slope_mean"]:.2f}({da[PLOT_VAR+"_slope_ci"]:.2f})")
    r_formatted = name.replace('_',' ').replace("nh","NH").replace("sh","SH") 
    r_formatted = r_formatted[0].upper() + r_formatted[1:]
    ax.set_title(r_formatted)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc=2,frameon=False,fontsize=10)
fig.supylabel(PLOT_VAR+" EEI / W m$^{-2}$")
plt.savefig(f"figures/regime_timeseries_{PLOT_VAR}.png",
        dpi=300, bbox_inches='tight')