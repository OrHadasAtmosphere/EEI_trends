import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe

np.seterr(divide="ignore", invalid="ignore")

from obs_io import to_trend, add_weights, march_to_feb_years, seasonal_means
from obs_io.clrsky_helper import T_strat, get_gammaLR, get_Trad_total, OLR_from_Tem

rh_var = "crh_600_400"

# load ERA5-drivers
ceres = xr.open_mfdataset(["pp/ceres_trends.nc"]).sel(season="ANN")
drivers = xr.open_mfdataset(["raw_data/drivers_levels.nc", "raw_data/drivers_levels_2026.nc", "raw_data/rh.nc"])
drivers = drivers.rename({'valid_time':"time", "latitude":"lat", "longitude":"lon"}).drop_vars(["number","expver"])
ceres = ceres.sortby(["lat","lon"])
regridder = xe.Regridder(drivers, ceres, method="bilinear")
drivers = regridder(drivers)
drivers = drivers.sel(time=slice("2000-03-01", "2026-03-01")).sel(lat=slice(-40,40))
drivers = drivers.chunk({
    "time": 12,
    "lat": 20,
    "lon": 20,
})
ts = drivers.skt
rh = drivers[rh_var]
del regridder, ceres, drivers

# load CO2
url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv"
df = pd.read_csv(url, comment="#")
df["time"] = pd.to_datetime(
    dict(year=df["year"], month=df["month"], day=1)
)
ds = (
    df.set_index("time")
      .to_xarray()
)[["average"]].rename({"average":"co2"})
ds = ds.sel(time=slice("2000-03-01", "2026-03-01"))
co2 = ds.co2 / 1e6 # ppm -> q
co2 = co2.chunk({"time": 12})
del ds

## do clear-sky calcs
nu = np.linspace(0.1, 1500, 300)
nu_da = xr.DataArray(
    nu,
    dims=["nu"],
    coords={"nu": nu},
    name="nu",
    attrs={"units": "cm^-1"},
)

meanTs = add_weights(ts).weighted(ts.days_in_month).mean("time").persist()
meanRH = add_weights(rh).weighted(rh.days_in_month).mean("time").persist()
meanco2 = add_weights(co2).weighted(co2.days_in_month).mean("time").persist()

def olr_timeseries_to_lwclr_trend(ds):
    lwclr_ts = seasonal_means(march_to_feb_years(add_weights(ds))) * -1
    return to_trend(lwclr_ts).olr

# co2-only
gammaLR = get_gammaLR(meanTs, T_strat)
Tem = get_Trad_total(nu_da, meanTs, T_strat, gammaLR, meanRH, co2)
OLR_co2 = OLR_from_Tem(nu_da, Tem)
lwclr_varco2 = olr_timeseries_to_lwclr_trend(OLR_co2.to_dataset(name="olr")).rename("lwclr_varco2").compute()
lwclr_varco2.attrs = {
    "long_name": (
        "CO2-component of reconstructed clear-sky LW trend"
    ),
    "units": "W m-2 decade-1",
}
del gammaLR, Tem, OLR_co2
print("done co2")

# Ts-only
gammaLR = get_gammaLR(ts, T_strat)
Tem = get_Trad_total(nu_da, ts, T_strat, gammaLR, meanRH, meanco2)
OLR_Ts = OLR_from_Tem(nu_da, Tem)

lwclr_varTs = olr_timeseries_to_lwclr_trend(OLR_Ts.to_dataset(name="olr")).rename("lwclr_varTs").compute()
lwclr_varTs.attrs = {
    "long_name": (
        "Ts-component of reconstructed clear-sky LW trend"
    ),
    "units": "W m-2 decade-1",
}
del gammaLR, Tem, OLR_Ts
print("done Ts")

# RH-only
gammaLR = get_gammaLR(meanTs, T_strat)
Tem = get_Trad_total(nu_da, meanTs, T_strat, gammaLR, rh, meanco2)
OLR_RH = OLR_from_Tem(nu_da, Tem)

lwclr_varRH = olr_timeseries_to_lwclr_trend(OLR_RH.to_dataset(name="olr")).rename("lwclr_varRH").compute()
lwclr_varRH.attrs = {
    "long_name": (
        "RH-component of reconstructed clear-sky LW trend"
    ),
    "units": "W m-2 decade-1",
}
del gammaLR, Tem, OLR_RH
print("done rh")

# all together
gammaLR = get_gammaLR(ts, T_strat)
Tem = get_Trad_total(nu_da, ts, T_strat, gammaLR, rh, co2)
OLR_all = OLR_from_Tem(nu_da, Tem)

lwclr_all = olr_timeseries_to_lwclr_trend(OLR_all.to_dataset(name="olr")).rename("lwclr_all").compute()
lwclr_all.attrs = {
    "long_name": (
        "total reconstructed clear-sky LW trend"
    ),
    "units": "W m-2 decade-1",
}
del gammaLR, Tem, OLR_all
print("done all")

# output
out = xr.Dataset({
    "lwclr_varRH": lwclr_varRH,
    "lwclr_varTs": lwclr_varTs,
    "lwclr_varco2": lwclr_varco2,
    "lwclr_all": lwclr_all,
})

out.attrs = {
    "description": (
        "Reconstructed clear-sky longwave trends "
        "from analytic, idealized spectral Koll et al (2023) and Czarnecki and Pincus (2026)"
    )
}
print(out)
out.to_netcdf("pp/analytic_lwclr_trends_"+rh_var+".nc")

