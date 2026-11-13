import numpy as np
import scipy.special

Rd = 287.0 # J/kg/K
Rv = 461.0 # J/kg/K
cp = 1004.0 # J/kg/K
Lv = 2.5e6 # J/kg
g = 9.8 # m/s2
T0 = 273.16 # K
e0 = 610.78 # Pa

def qsat(T,p):
    psat = e0 * np.exp(-Lv/Rv * (1/T - 1/T0))
    return Rd/Rv * psat / (p - psat)

def pres(z, ps):
    return ps * np.exp((-g*z) / (Rd*T0))

def Gm(T,p):
    rv = qsat(T,p) / (1 - qsat(T,p))
    Gd = (g/cp)
    return Gd * (1 + (Lv*rv)/(Rd*T)) / (1 + (Lv**2*rv)/(Rv*cp*T**2))

def lcl(p,T,rh):
    # Parameters
    E0v   = 2.3740e6   # J/kg
    cva   = 719        # J/kg/K
    cvv   = 1418       # J/kg/K 
    cvl   = 4119       # J/kg/K 
    cpa   = cva + Rd
    cpv   = cvv + Rv

    # The saturation vapor pressure over liquid water
    def pvstarl(T):
        return e0 * (T/T0)**((cpv-cvl)/Rv) * \
            np.exp( (E0v - (cvv-cvl)*T0) / Rv * (1/T0 - 1/T) )

    pv = rh * pvstarl(T)
    qv = Rd*pv / (Rv*p + (Rd-Rv)*pv)
    rgasm = (1-qv)*Rd + qv*Rv
    cpm = (1-qv)*cpa + qv*cpv
    if rh == 0:
        return cpm*T/g
    
    aL = -(cpv-cvl)/Rv + cpm/rgasm
    bL = -(E0v-(cvv-cvl)*T0)/(Rv*T)
    cL = pv/pvstarl(T)*np.exp(-(E0v-(cvv-cvl)*T0)/(Rv*T))
    zlcl = cpm*T/g*( 1 - \
        bL/(aL*scipy.special.lambertw(bL/aL*cL**(1/aL),-1).real) )

    return zlcl

def calc_EIS(Ts, ps, T700, T850):
    th700 = T700 * (ps / 700e2)**(Rd/cp)
    LTS = th700 - Ts
    z700 = (Rd*Ts / g) * np.log(ps / 700e2)
    Gm_850 = Gm(T850, 850e2)
    
    RHs = 0.8
    zLCL = lcl(ps, Ts, RHs)

    return LTS - Gm_850*(z700 - zLCL)

