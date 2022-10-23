from tapper.field import *
import pandas as pd
import xarray as xr
import numpy as np
from copy import deepcopy
from scipy.optimize import leastsq, curve_fit
import os


def doifft(Tmax=1, N=1, F=lambda s: 0):
    dw = np.pi / Tmax
    aa = np.zeros(2 * N, dtype=np.complex128)
    for k in range(N):
        z = F(1e-20 + k * dw * 1j)
        aa[k] = z
        if k > 0:
            aa[2 * N - k] = np.conj(z)
    bb = np.fft.ifft(aa)
    bb = np.real(bb) / np.pi * dw * N
    return bb

def universal(t, pars, opts, eps, L, D, model='diff'):

    Tmax = t[-1] + t[1]
    N = len(t)
    dt = t[1]

    def r_s_rev_ads(s, pars, opts, eps, L, D):
        Sv = opts[0] #Either true surface ares m^2_s/m^3_s or density kg/m^3_s
        a_s = opts[1] #Either true surface site density mol/m^2_s or concentration mol/kg
        ka = pars[0]
        kd = pars[1]
        rs = s*(1.0-eps)*Sv*a_s*ka/(s+kd)
        return rs

    def r_s_pore_eq_lin(s, pars, opts, eps, L, D):
        K_H = pars[0]
        Dp = pars[1]
        Sv = opts[0] #True surface ares m^2_s/m^3_s        
        R = opts[1]
        eps_p = opts[2]
        tau_b = eps*L**2.0/D
        tau_p = eps_p*R**2.0/Dp
        alpha = tau_b/tau_p
#         rs = (1.0-eps)*Sv*Dp*(s/alpha)**0.5*K_H*np.tanh((s/alpha)**0.5)/R
        rs = (1.0-eps)*Sv*(s*eps_p*Dp)**0.5*K_H*np.tanh(R*(s*eps_p/Dp)**0.5)
        return rs 

    def r_s_pore_eq_sph(s, pars, opts, eps, L, D):
        K_H = pars[0]
        Dp = pars[1]
        Sv = opts[0]        
        R = opts[1]
        eps_p = opts[2]        
        tau_b = eps*L**2.0/D
        tau_p = eps_p*R**2.0/Dp
        alpha = tau_b/tau_p
        rs = (1.0-eps)*Sv*Dp*K_H*((s/alpha)**0.5/np.tanh((s/alpha)**0.5) - 1.0)/R
        #rs = (1.0-eps)*Sv*Dp*K_H**((s*eps_p/Dp)**0.5/np.tanh(R*(s*eps_p/Dp)**0.5)+1.0/R)
        return rs 

    def r_s_pore_ads_lin(s, pars, opts, eps, L, D):
        ka = pars[0]
        kd = pars[1]
        k_ads = pars[2]
        k_des = pars[3]
        Dp = pars[4]
        Sv = opts[0]        
        R = opts[1]
        eps_p = opts[2]
        as_p = opts[3]        
        Phi = (eps_p*s+(1.0-eps_p)*as_p*ka-(1.0-eps_p)*as_p*kd*ka/(s+kd))/Dp
        rs = (1-eps)*Sv*k_ads*Phi**0.5*np.tanh(Phi**0.5*R)/\
             (k_des+Dp*Phi**0.5*np.tanh(Phi**0.5*R))
        return rs
    def r_s_surf_bar_lin_pore_full(s, pars, opts, eps, L, D):
        ka = pars[0]
        kd = pars[1]
        Dp = pars[-1]
        k_ent = pars[2]
        k_ext = pars[3]
        Sv = opts[0]
        a_s = opts[1]
        R = opts[2]
        eps_p = opts[3]
        Gamma = (eps_p*s/Dp)**0.5
        phi = k_ent * Dp * Sv * Gamma / (Gamma * Dp * Sv + k_ext * eps_p / np.tanh(Gamma*R))
        rs = a_s*Sv*(1-eps)*ka*((s+phi)/(s+kd+phi))
        return rs
    def r_s_surf_bar_lin_pore_part(s, pars, opts, eps, L, D):
        ka = opts[4]
        kd = opts[5]
        Dp = pars[-1]
        k_ent = pars[0]
        k_ext = pars[1]
        Sv = opts[0]
        a_s = opts[1]
        R = opts[2]
        eps_p = opts[3]
        Gamma = (eps_p*s/Dp)**0.5
        phi = k_ent * Dp * Sv * Gamma / (Gamma * Dp * Sv + k_ext * eps_p / np.tanh(Gamma*R))
        rs = a_s*Sv*(1-eps)*ka*((s+phi)/(s+kd+phi))
    def r_s_SBLP_int_ads(s, pars, opts, eps, L, D):
        ka = pars[0]
        kd = pars[1]
        k_ent = pars[2]
        k_ext = pars[3]
        ka_int = pars[4]
        kd_int = pars[5]
        Dp = pars[-1]
        Sv = opts[0]
        a_s = opts[1]
        R = opts[2]
        eps_p=opts[3]
        a_s_int = opts[4]
        Gamma = (s/Dp*(eps_p + a_s_int * ka_int / (s + kd_int)))**0.5
        phi = k_ent * Dp * Sv * Gamma / (Gamma * Dp * Sv + k_ext * eps_p / np.tanh(Gamma*R))
        rs = a_s*Sv*(1-eps)*ka*((s+k_ent/(1+k_ext*Beta))/(s+kd+k_ent/(1+k_ext*Beta)))
        return rs
    def r_s_SBLP_int_ads_part(s, pars, opts, eps, L, D):
        ka = opts[5]
        kd = opts[6]
        k_ent = pars[0]
        k_ext = pars[1]
        ka_int = pars[2]
        kd_int = pars[3]
        Dp = pars[-1]
        Sv = opts[0]
        a_s = opts[1]
        R = opts[2]
        eps_p=opts[3]
        a_s_int = opts[4]
        Gamma = (s/Dp*(eps_p + a_s_int * ka_int / (s + kd_int)))**0.5
        phi = k_ent * Dp * Sv * Gamma / (Gamma * Dp * Sv + k_ext * eps_p / np.tanh(Gamma*R))
        rs = a_s*Sv*(1-eps)*ka*((s+phi)/(s+kd+phi))
        return rs
        
    if model == 'Diff_3Z':
        rs = None    
    elif model == 'rev_ads':
        rs = r_s_rev_ads
    elif model == 'pore_eq_lin':
        rs = r_s_pore_eq_lin
    elif model == 'pore_eq_sph':
        rs = r_s_pore_eq_sph        
    elif model == 'pore_ads_lin':
        rs = r_s_pore_ads_lin
    elif model == 'surf_bar_full':
        rs = r_s_surf_bar_lin_pore_full
    elif model == 'surf_bar_part':
        rs = r_s_surf_bar_lin_pore_part
    elif model == 'SBIALP_full':
        rs = r_s_SBLP_int_ads
    elif model == 'SBIALP_part':
        rs = r_s_SBLP_int_ads_part
    else:
        raise KeyError('Unknown model!')

    def unimatrix(s=1, rs=None, pars=0, opts=0, eps=1, L=1, D=1):
        if rs is None:
            z = np.sqrt((s * eps) / D)
        else:
            z = np.sqrt((s * eps + rs(s, pars, opts, eps, L, D)) / D)

        cz = np.cosh(z * L)
        sz = np.sinh(z * L)
        M = np.matrix([[cz, sz / (D * z)], [D * z * sz, cz]])
        return M


    # Matrix solution
    def F(s):
        if len(L)==3:
            M = np.dot(
                       np.dot(
                              unimatrix(L=L[0], eps=eps[0], D=D[0], s=s), 
                              unimatrix(L=L[1], eps=eps[1], D=D[1], rs=rs, 
                                        pars=pars, s=s, opts=opts)
                              ), 
                       unimatrix(L=L[2], eps=eps[2], D=D[2], s=s)
                       )
        elif len(L)==1:
            M = unimatrix(L=L[0], eps=eps[0], D=D[0], rs=rs, 
                                        pars=pars, s=s, opts=opts)


        return 1.0 / M[1, 1]
        
    Fmod = doifft(Tmax=Tmax, N=N, F=F)

    return Fmod[:len(t)]

def pulse(pars, opts, model, T=1, dt=0.001, Nx=1000):
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)

    Temp = opts['T']
    Temp_ref = opts['Tref']
    Mref = opts['Mref']
    M = opts['M']
    Sw = opts['Surface per weight']
    mass = opts['Sample mass']
    eps = [opts['porosity'],opts['porosity'],opts['porosity']]
    eps_p=opts['microporosity']

    Area = opts['cross_section']
    R = opts['micropore_Rp']
#     print((Temp*Mref/M/Temp_ref)**0.5)
    Dif = np.array(opts['Dref zones'])*(Temp*Mref/M/Temp_ref)**0.5

    Length = opts['length']
    Sv = Sw*mass/Area/Length[1]/(1-eps[1])
    ac = opts['AS concentration'] / Sw
#     print("Sv*ac = ", Sv*ac)
    tb = eps[1] *Length[2]**2 / Dif[1]
#     tl=t/tb
#     print(Dif)
    try:
        a_s_int = opts["Internal AS concentration"]
    except:
        pass
    if model == 'Diff_3Z':
        params=[0,0]
        options=[0,0]
    elif model == 'rev_ads':
        params = pars
        options = [Sv, ac]
    elif model == 'pore_eq_lin':
        params = pars
        options = [Sv, R, eps_p]
    elif model == 'pore_eq_sph':
        params = pars
        options = [Sv, R, eps_p]
#     elif model == 'pore_ads_lin':
#         params = pars
#         options = [Sv, R, eps_p]
    elif model == 'surf_bar_full':
        params = pars
        options = [Sv, ac, R, eps_p]
    elif model == 'surf_bar_part':
        assert opts['Surface adsorption/desorption constants'] is not None
        ka, kd = opts['Surface adsorption/desorption constants']
        params = pars
        options = [Sv, ac, R, eps_p, ka, kd]
    elif model == 'SBIALP_full':
        params = pars
        assert a_s_int is not None
        options = [Sv, ac, R, eps_p, a_s_int]
    elif model == 'SBIALP_part':
        assert a_s_int is not None
        assert opts['Surface adsorption/desorption constants'] is not None
        ka, kd = opts['Surface adsorption/desorption constants']
        params = pars
        options = [Sv, ac, R, eps_p, a_s_int, ka, kd]
    else:
        raise KeyError('Unknown model!')
#     print(params)    
    y = universal(t, params, options, eps, L=Length, D=Dif, model=model)
    y[:]=y[:]-y[0]
    numpy_array = np.array((t, np.asarray(y)))
    
    return numpy_array

