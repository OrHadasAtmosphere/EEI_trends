### helper functions for radiation calculations

import numpy as np

import scipy as sp
from scipy.constants import speed_of_light as c
import scipy.constants
from scipy.constants import sigma as SB_sigma
import metpy.calc as mpcalc
from metpy.units import units


from importlib import reload

# create idealized moist-adiabatic profile
def create_profile_moist(T_s, rh, p):
    ''' 
    create an idealized profile with given surface temperature that follows
    a moist adiabatic lapse rate until a 200K isothermal stratosphere

    IN:
        T_s [K]: given surface temperature
        rh [%]: given relative humidity, constant throughout atmosphere
        
    OUT:
        T_profile [K]: temperature profile
        vmr [unitless]: water vapor volume mixing ratio
        q_star_s [kg/kg]: saturation mass mixing ratio of water vapor at the surface        
    
    '''
    T_s = units.Quantity(T_s, 'K')
    # create pressure, temperature, and mixing ratio profiles for each initial temp
    pressure_profile = units.Quantity(p, 'Pa')
    pressure_profile_temp = p[np.where(p > 100)[0]]
    pressure_profile_temp = units.Quantity(pressure_profile_temp, 'Pa')
    p_s = units.Quantity(p[0], 'Pa')

    temperature = mpcalc.moist_lapse(pressure_profile_temp, T_s, p_s)
    idx = np.where(temperature > units.Quantity(200, 'K'))[0]
    T_profile = units.Quantity(200, 'K')*np.ones(len(p))
    T_profile[:idx[-1] + 1] = temperature[idx]

    mass_mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(pressure_profile, T_profile, rh).to('kg/kg') 
    q_star_s = mpcalc.saturation_mixing_ratio(p_s, T_profile[0]).to('kg/kg')

    # mmr = vmr * molar mass / molar mass of dry air
    vmr = mass_mixing_ratio * (29 / 18.015)       
    
    return T_profile, vmr, q_star_s


# Planck function
def B_nu(T, nu):
    """
    Planck function
    
    In:
    T [K]: temperature
    nu [cm-1]: multiply all nu's by 100 to convert to m-1 in formula, then multiply
               B by 100 for units of W/m^2/sr/cm-1
        
    Out:
        Planck function in units of W/m^2/sr/cm-1
    """
    k_B = scipy.constants.k # Boltzmann constant
    h = scipy.constants.h # Planck constant
    c = scipy.constants.speed_of_light # speed of light in a vacuum
    
    return ((2*h*(c**2)*((100*nu)**3))/(np.exp((h*c*(100*nu))/(k_B*T)) - 1))*100

# draw idealized exponential absorption coefficient
def k(k_0, nu_0, l, wavenumbers):
    """
    calculate idealized exponential absorption coefficient
    
    In:
        k_0 [m^2/kg]: height of peak of the absorption band
        nu_0 [cm-1]: center of the band or where peak is, in wavenumber
        l [cm-1]: exponential slope of the coefficient
        wavenumbers [cm-1]: wavenumber range across which to calculate

    Out:
        k [m^2/kg]
    """
    # return k_0 * np.exp(-(np.abs(wavenumbers - nu_0*np.ones(len(wavenumbers)))/l))
    return k_0 * np.exp(-(np.abs(wavenumbers - nu_0)/l))



### Clear-sky functions from Koll et al., 2023
### Source: https://github.com/danielkoll/spectral_feedbacks/tree/main

### spectral/atmospheric constants
nu = np.linspace(0.1, 1500, 15000) # typical longwave spectrum

T_strat = 200 # K, stratospheric temperature

# co2
qco2 = 0.0004 # CO2 concentration (molar/vmr)
l_co2 = 10.2 # cm-1
kappa_co2 = 500 # m2/kg

# h2o rot band
ln0 = 58 # cm-1
kappa0 = 165 # m2/kg

## h2o v-r band
ln1 = 60 # cm-1
kappa1 = 15 # m2/kg

## h2o continuum
lX = 275   # cm-1
kappaX0 = 0.1 # m2/kg
kappaX = k(kappaX0, 700, lX, nu)
cntm_alpha = 7 # continuum dependence with temperature


### Thermodynamics
H2O_TriplePointT = 2.731500e+02
H2O_TriplePointP = 6.110000e+02
N_avogadro = 6.022136736e23  #Avogadro's number
k_B = scipy.constants.k # Boltzmann constant
R_star = 1000.*k_B*N_avogadro   #Universal gas constant
Rv = R_star/1.800000e+01
Lvap = 2.493000e+06

esat = lambda T: get_satvps(T, H2O_TriplePointT, H2O_TriplePointP, Rv, Lvap)  # saturation vapor pressure

ps_dry = 1e5 #Pa; reference pressure
g = 9.81 #m/s2; gravity
D = 1/2 # two-stream hemispheric integration
R_co2 = R_star/4.400000e+01 # CO2 gas constant
R = R_star/28.97 # gas constant
cp_const = 1004.67 # J/(kg*K) -- constant pressure specific heat for dry air


def get_kappa_co2(nu, p):
    """
    calculate idealized exponential absorption coefficient for CO2

    In:
        nu [cm-1]: wavenumber array
        p [Pa]: pressure
    Out:
        kappa [m^2/kg]: absorption coefficient
    """
    p0 = 1e5 # Pa; reference pressure
    n0 = 667.5 # cm-1; band center

    kappa = kappa_co2 * (p/p0) * np.exp( -np.abs(nu-n0)/l_co2 )

    return kappa

def get_kappa_h2o(nu, p):
    """
    calculate idealized exponential absorption coefficient for H2O line absorption
    
    In:
        nu [cm-1]: wavenumber array
        p [Pa]: pressure
    Out:
        kappa [m^2/kg]: absorption coefficient
    """
    n0, n1 = 150, 1500 #cm-1; band centers
    p0 = 1e5 #Pa; reference pressure

    kappa = np.maximum( kappa0*(p/p0)*np.exp( -np.abs(nu-n0)/ln0 ), kappa1*(p/p0)*np.exp( -np.abs(nu-n1)/ln1 ) ) 

    return kappa 

def get_satvps(T, T0, e0, Rv, Lv):
    """
    saturation vapor pressure from Clausius-Clapeyron relation

    In:
        T [K]: temperature
        T0 [K]: reference temperature
        e0 [Pa]: reference saturation vapor pressure
        Rv [J/(kg*K)]: gas constant for water vapor
        Lv [J/kg]: latent heat of vaporization
    Out:
        e_s [Pa]: saturation vapor pressure
    """
    return e0*np.exp(-(Lv/Rv)*(1./T - 1./T0))

def get_kappa_selfcont(nu, T, RH=1., T0=300.):
    """
    calculate idealized exponential absorption coefficient for H2O self-continuum absorption

    In:
        nu [cm-1]: wavenumber array
        T [K]: temperature
        RH [-]: reference relative humidity (0-1)
        T0 [K]: reference temperature
    Out:
        kappa [m^2/kg]: absorption coefficient
    """
    kappaX = k(kappaX0, 700, lX, nu) # exponential continuum

    TX =  300 # K; reference temperature
    esatX = esat(TX) # calculate saturation vapor pressure
    
    gammaWV = Lvap/(Rv*T0)
    esat0 = esat(T0)
    e = RH *esat0 * (T/T0)**gammaWV

    kappa0 = kappaX * (esat0/esatX) * (T0/TX)**(-cntm_alpha) #  if T0 for CC approx differs from cntm_spec.T0: properly rescale kappa0, so kappa0=kappa0(T0)    
    kappa = kappa0 * (e/esat0) * (T/T0)**(-cntm_alpha) #* np.ones_like(nu)

    return kappa


def convert_molar_to_mass_ratio(molar_i, R_i, R_air):
    """
    convert molar mixing ratio to mass mixing ratio 
    In:
        molar_i [-]: molar mixing ratio of species i
        R_i [J/(kg*K)]: gas constant of species i
        R_air [J/(kg*K)]: gas constant of air
    Out:
        q_i [-]: mass mixing ratio of species i
    """
    molar_air = 1. - molar_i
    R_mean = 1./(molar_i/R_i + molar_air/R_air)
    q_i = molar_i * R_mean/R_i
    return q_i

# Absorption due to CO2 line wings
def get_Trad_co2(nu, Ts, gammaLR, qco2):
    """
    calculate radiating temperature due to CO2 line absorption 
    In:
        nu [cm-1]: wavenumber array
        Ts [K]: surface temperature
        gammaLR [K/km]: bulk atmospheric lapse rate
        qco2 [-]: molar mixing ratio of CO2
    Out:
        Trad [K]: radiating temperature
    """
    tau0 = get_kappa_co2(nu, ps_dry) * ps_dry/(2.*g*D)
    qco2 = convert_molar_to_mass_ratio(qco2, R_co2, R)
    Trad = Ts * ( 1./(tau0 * qco2)  )**(gammaLR/2.)
    return Trad

# Absorption due to H2O line wings
def get_Trad_h2o(nu, Ts, gammaLR, RH, T0=300.):
    """
    calculate radiating temperature due to H2O line absorption
    In:
        nu [cm-1]: wavenumber array
        Ts [K]: surface temperature
        gammaLR [K/km]: bulk atmospheric lapse rate
        RH [-]: relative humidity (0-1)
        T0 [K]: reference temperature
    Out:
        Trad [K]: radiating temperature
    """
    gammaWV = Lvap/(Rv * T0)
    kappa = get_kappa_h2o(nu, ps_dry)
    esat0 = esat(T0)
    tau_a = R/Rv * kappa*esat0/(g*D)
    Trad = T0 * ( (1.+gammaWV*gammaLR)/(tau_a * RH)  )**(gammaLR/(1.+gammaWV*gammaLR)) * (Ts/T0)**(1./(1.+gammaWV*gammaLR))
    return Trad

# Absorption due to H2O self-continuum
def get_Trad_cntm(nu, gammaLR, RH, T0=300.):
    """
    calculate radiating temperature due to H2O self-continuum absorption
    In:
        nu [cm-1]: wavenumber array
        gammaLR [K/km]: bulk atmospheric lapse rate
        RH [-]: relative humidity (0-1)
        T0 [K]: reference temperature
    Out:
        Trad [K]: radiating temperature
    """
    gammaWV = Lvap/(Rv * T0)
    kappa0 = get_kappa_selfcont(nu, T0, RH=1., T0=T0)  # here: kappa0 defined at RH=1,T=T0
    esat0 = esat(T0)    
    tau_b = R/Rv * kappa0*esat0/(g*D)
    Trad = T0 * ( ((2.*gammaWV-cntm_alpha)*gammaLR)/(tau_b * RH**2))**(1./(2.*gammaWV-cntm_alpha))
    return Trad

# Overall radiating temp
def get_Trad_total(nu, Ts, Tstrat, gammaLR, RH, qco2, T0=300.):
    """
    calculate overall radiating temperature
    -> defined as minimum at each wavenumber
    In:
        nu [cm-1]: wavenumber array
        Ts [K]: surface temperature
        Tstrat [K]: stratospheric temperature
        gammaLR [K/km]: bulk atmospheric lapse rate
        RH [-]: relative humidity (0-1)
        qco2 [-]: molar mixing ratio of CO2
        T0 [K]: reference temperature
    Out:
        Trad [K]: radiating temperature
    """
    Tco2 = get_Trad_co2(nu, Ts, gammaLR, qco2)
    Th2o = get_Trad_h2o(nu, Ts, gammaLR, RH, T0=T0)
    Tcntm = get_Trad_cntm(nu, gammaLR, RH, T0=T0)        

    Tatm = np.minimum(np.minimum(Tco2,Th2o),Tcntm)
    Trad = np.maximum(np.minimum(Ts,Tatm),Tstrat)
    return Trad


# Estimate gammaLR
def get_gammaLR(Ts, Tstrat):
    """
    estimate bulk atmospheric lapse rate
    In:
        Ts [K]: surface temperature
        Tstrat [K]: stratospheric temperature
    Out:
        gammaLR [K/km]: bulk atmospheric lapse rate
    """
    qsat_surf = (esat(Ts)/Rv) / (ps_dry/R + esat(Ts)/Rv) # non-dilute: use this when comparing with full LR calcs
    Tavg = 0.5*(Ts+Tstrat)
    gammaLR = R*Tavg*np.log(Ts/Tstrat) / (cp_const * (Ts-Tstrat) + Lvap*qsat_surf)
    return gammaLR


def OLR_from_Tem(nu, T_em):
    """
    Calculate OLR from emission temperature
    In:
        nu [cm-1]: wavenumber array
        T_em [K]: emission temperature
    Out:
        int_OLR [W/m^2]: integrated OLR
    """
    return (np.pi*B_nu(T_em, nu)).integrate("nu")
