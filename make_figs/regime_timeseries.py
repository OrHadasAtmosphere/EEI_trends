import xarray as xr
import matplotlib.pyplot as plt

PLOT_VAR = "net"

ds = xr.open_dataset("pp/regime_mean_trends.nc")
fig,axes=plt.subplots(2,4,figsize=(12,4),sharex=True,constrained_layout=True)
for i,r in enumerate(ds.regime):
    ax=axes.flatten()[i]
    da = ds.sel(season="ANN", regime=r)
    da[PLOT_VAR].plot(ax=ax)
    da[PLOT_VAR+"_fit"].plot(ax=ax, label=f"{da[PLOT_VAR+"_slope_mean"]:.2f}({da[PLOT_VAR+"_slope_ci"]:.2f})")
    ax.set_title(r.values)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc=2,frameon=False,fontsize=10)
fig.supylabel(PLOT_VAR+" EEI / W m$^{-2}$")
plt.savefig(f"figures/regime_timeseries_{PLOT_VAR}.png",
        dpi=300, bbox_inches='tight')