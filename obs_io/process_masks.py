from . import regimes_from_clim

### 
# make regime masks
# (1990-2000 climatology)
###

regimes_from_clim("era5_clim.nc", "regime_masks.nc")
