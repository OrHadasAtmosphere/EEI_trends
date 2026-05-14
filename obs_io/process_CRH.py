import numpy as np
import xarray as xr

g = 9.80665       # m s-2
T0  = 273.15      # K
e0  = 611.2       # Pa
Lv  = 2.5e6       # J kg-1
Rv  = 461.5       # J kg-1 K-1
eps = 0.622       # Rd/Rv

def es(T):
    return e0 * np.exp(
        (Lv / Rv) * (1.0 / T0 - 1.0 / T)
    )
def qs(T, p):
    return eps * es(T) / (p - (1.0 - eps) * es(T))

ds = xr.open_mfdataset(["raw_data/3D_RH.nc"])
ds = ds.rename({"pressure_level":"plev"}).drop_vars(["number","expver"])

# interpolate to mid points in logp
ds = ds.sortby("plev", ascending=False)
ds = ds.sel(plev=slice(1000, 300))
p = ds.plev.values  # hPa
logp = np.log(p)
logp_mid = 0.5 * (logp[:-1] + logp[1:])
p_mid = np.exp(logp_mid)  # hPa

q_mid = (
    ds.q
    .assign_coords(logplev=("plev", logp))
    .swap_dims({"plev": "logplev"})
    .interp(logplev=logp_mid)
    .swap_dims({"logplev": "plev"})
)

T_mid = (
    ds.t
    .assign_coords(logplev=("plev", logp))
    .swap_dims({"plev": "logplev"})
    .interp(logplev=logp_mid)
    .swap_dims({"logplev": "plev"})
)

q_mid = q_mid.assign_coords(plev=p_mid)
T_mid = T_mid.assign_coords(plev=p_mid)
qs_mid = qs(T_mid, T_mid.plev*100) # Pa

# integrate by dp
dp = (p[:-1] - p[1:]) * 100.0  # Pa
dp_da = xr.DataArray(
    dp,
    dims=["plev"],
    coords={"plev": q_mid.plev.values},
)

wv_mass = ((q_mid * dp_da) / g).sum(dim="plev")
sat_mass = ((qs_mid * dp_da) / g).sum(dim="plev")

column_rh = wv_mass / sat_mass
column_rh.name = "column_rh"
column_rh.attrs = {
    "long_name": "Column Relative Humidity",
    "description": (
        "Column water vapor mass divided by "
        "column saturated water vapor mass"
    ),
    "upper_boundary": "300 hPa",
    "units": "1",
}

column_rh.to_netcdf("raw_data/column_rh.nc")
