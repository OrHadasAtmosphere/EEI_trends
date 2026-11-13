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

ds = xr.open_mfdataset(["raw_data/3D_RH.nc", "raw_data/3D_RH_2026.nc"])
ds = ds.rename({"pressure_level":"plev"}).drop_vars(["number","expver"])
ds = ds.sortby("plev", ascending=False)

def get_weighted_avg(ds, pmin=300, pmax=1000):
    # interpolate to mid points in logp
    da = ds.sel(plev=slice(pmax, pmin))
    p = da.plev.values  # hPa
    logp = np.log(p)
    logp_mid = 0.5 * (logp[:-1] + logp[1:])
    p_mid = np.exp(logp_mid)  # hPa
    
    q_mid = (
        da.q
        .assign_coords(logplev=("plev", logp))
        .swap_dims({"plev": "logplev"})
        .interp(logplev=logp_mid)
        .swap_dims({"logplev": "plev"})
    )
    
    T_mid = (
        da.t
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
    
    crh = (wv_mass / sat_mass).rename(f"crh_{pmax}_{pmin}")
    crh.attrs = {
        "long_name": "Column Relative Humidity",
        "description": (
            "Column water vapor mass divided by "
            "column saturated water vapor mass"
        ),
        "upper_boundary": f"{pmin} hPa",
        "lower boundary": f"{pmax} hPa",
        "units": "1",
    }
    return crh

def get_single_lev(ds, pi=400):
    dsi = ds.sel(plev=pi, method="nearest")
    rhi = (
        dsi.q / qs(dsi.t, dsi.plev*100) # Pa
    ).rename("rh400").drop_vars(["plev"])
    rhi.attrs = {
        "long_name": "Relative Humidity at 400 hPa",
        "description": (
            "Specific humidity(p400) / Saturation humidity(T400,p400)"
        ),
        "units": "1",
    }
    return rhi

crh_1000_300 = get_weighted_avg(ds, pmax=1000, pmin=300)
crh_800_300 = get_weighted_avg(ds, pmax=800, pmin=300)
crh_600_400 = get_weighted_avg(ds, pmax=600, pmin=400)
rh400 = get_single_lev(ds, pi=400)
rh500 = get_single_lev(ds, pi=500)

rh = xr.Dataset(
    data_vars={
        "crh_1000_300": crh_1000_300,
        "crh_800_300": crh_800_300,
        "crh_600_400": crh_600_400,
        "rh400": rh400,
        "rh500": rh500,
    },
)

rh.to_netcdf("raw_data/rh.nc")
