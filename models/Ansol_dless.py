import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sympy import sympify
from .thermodynamics.desorb_const import k_des_frenkel
from .thermodynamics.desorb_const import k_des_frenkel_manual
from .thermodynamics.adsorb_const import ka_hertz_knud
import mpmath as mp


def analytical_adsdes(pars, opts, T, dt, Nx):
    Temp = opts['T']                                    # Temperature, K
    Temp_ref = opts['Tref']                              # Reference temperature for Knudsen diffusivity, K
    Length = opts['length']
#     L = Length[2] - Length[1]                              # List of zone lengths, m 
    L = Length[-1]
    #     dx = opts['dx']                                  # Reactor-scale grid step, m
    eps = opts['porosity']                           # List of zone porosities
    
    D_ref = opts['Dref zones'][0]                     # Reference Knudsen diffusivity, m^2/s
    Mref = opts['Mref']                              # Reference molecular mass for Knudsen diffusivity, g/mol
    M = opts['M']                                    # Molecular mass of the diffusing gas, g/mol

    Np = opts['Np']                                  # Pulse size, mol
    Sw = opts['Surface per weight']                        # Surface-to-volume ratio for the microporous particles, 1/m
    AS = opts['AS concentration']
    mass = opts['Sample mass'] 
    ac = AS/Sw
    
    D = D_ref*(Temp*Mref/M/Temp_ref)**0.5
    print(D)
    ka = pars[0]
    kd = pars[1]
    print(ka, kd)
    eps=opts['porosity']
    Area = opts['cross_section']
    dx = L/Nx
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)
    x = np.linspace(0, L, Nx+1)
    
    tau = t * D / eps / L**2
    C=0
    Fl=0
    n=10000
    
    Sv = Sw*mass/Area/L/(1-eps)
    print('Sv*As*(1-eps) = ', Sv*ac*(1-eps))
    kad = ac*Sv*(1-eps)*ka*L**2/D
    kdd = kd*eps*L**2/D

    for i in range(n):
        pn=(i+0.5)*np.pi
        rp = (-(pn**2+kad+kdd) + ((pn**2+kad+kdd)**2-4*pn**2*kdd)**(1/2))/2
        rm = (-(pn**2+kad+kdd) - ((pn**2+kad+kdd)**2-4*pn**2*kdd)**(1/2))/2
        An = (rp+pn**2+kad)/(rp-rm)
        Fl += (-1)**i*(2*i+1)*(An*np.exp(rm*tau) + (1-An)*np.exp(rp*tau))

    Fl=Fl*np.pi
    Fl[Fl<0]=0
    Flow=Fl*Np*D/eps/L**2
    df=pd.DataFrame({'time': t, 'Flow': np.asarray(Flow)})

    return df

def analytical_7P(pars, opts, T, dt=0.001, Nx=1000, inversion = 'ILap'):
    
    
    ka = pars[0]
    kd = pars[1]
    k_ent = pars[2]
    k_ext = pars[3]
    ka_int = pars[4]
    kd_int = pars[5]
    alpha = pars[-1]
    beta = opts[0]
    
    gamma = lambda s: (s/alpha*(1 + ka_int/(s/alpha+kd_int)))**0.5
    phi = lambda s: k_ent/(1+k_ext/(alpha * beta * gamma(s) * mp.tanh(gamma(s))))
    f = lambda s: ka*(s+phi(s))/(s+kd+phi(s))
#     Fp = 1 / mp.cosh((f+s)**0.5)
    
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)

    
    
    fp = lambda s: 1 / mp.cosh((f(s)+s)**0.5)

    if inversion == "ILap":
        Fl = map( lambda x: mp.invertlaplace(fp, x, method = 'talbot', dps = 5, degree = 5), t[1:])
        Flow = [0]
        Flow.extend(Fl)
        Flow = np.array(sympify(list(Flow)))
    elif inversion == 'ifft':
        
        Flow = doifft(Tmax=T, N = Nt+1, F = fp)[:Nt+1]
        
    Flow[Flow<0] = 0       
    numpy_array = np.array((t, np.asarray(Flow)), dtype = float)

    return numpy_array

def analytical_EqMP(pars, opts, T, dt=0.001, Nx=1000, inversion = 'ILap'):
    
    
    KH = pars[0]
    alpha = pars[-1]
    beta = opts[0]
    eps = opts[1]
    eps_p = opts[2]
    
    P = KH * (1-eps)*eps_p/eps
    
    gamma = lambda s: (s/alpha)**0.5
    f = lambda s: alpha*beta*P*gamma(s)*mp.tanh(gamma(s))
#     Fp = 1 / mp.cosh((f+s)**0.5)
    
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1) 

    fp = lambda s: 1 / mp.cosh((f(s)+s)**0.5)

    if inversion == "ILap":
        Fl = map( lambda x: mp.invertlaplace(fp, x, method = 'talbot', dps = 5, degree = 5), t[1:])
        Flow = [0]
        Flow.extend(Fl)
#         for i in t[1:]:
#             Fl = mp.invertlaplace(fp, i, method='talbot')
#             Fl = sympify(Fl)
#             Flow.append(Fl)
        Flow = np.array(sympify(list(Flow)))
    elif inversion == 'ifft':
        
        Flow = doifft(Tmax=T, N = Nt+1, F = fp)[:Nt+1]
        
    Flow[Flow<0] = 0       
    numpy_array = np.array((t, np.asarray(Flow)), dtype=float)

    return numpy_array

def analytical_DAMP(pars, opts, T, dt=0.001, Nx=1000, inversion = 'ILap'):
    
    
    ka = pars[0]
    kd = pars[1]
    alpha = pars[-1]
    R = opts[0]
    dl = opts[1]
    
    delta = R/dl
    
    gamma = lambda s: (s/alpha)**0.5
    phi = lambda s: alpha * delta * gamma(s)*mp.tanh(gamma(s))
    f = lambda s: ka * (s+phi(s))/(s + kd + phi(s))
#     Fp = 1 / mp.cosh((f+s)**0.5)
    
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)
 

    fp = lambda s: 1 / mp.cosh((f(s)+s)**0.5)

    if inversion == "ILap":
        Fl = map( lambda x: mp.invertlaplace(fp, x, method = 'talbot', dps = 5, degree = 5), t[1:])
        Flow = [0]
        Flow.extend(Fl)
#         for i in t[1:]:
#             Fl = mp.invertlaplace(fp, i, method='talbot')
#             Fl = sympify(Fl)
#             Flow.append(Fl)
        Flow = np.array(sympify(list(Flow)))
    elif inversion == 'ifft':
        
        Flow = doifft(Tmax=T, N = Nt+1, F = fp)[:Nt+1]
        
    Flow[Flow<0] = 0    
    numpy_array = np.array((t, np.asarray(Flow)), dtype=float)

    return numpy_array

def analytical_DAMPint(pars, opts, T, dt=0.001, Nx=1000, inversion = 'ILap'):
    
    
    ka = pars[0]
    kd = pars[1]
    ka_int = pars[2]
    kd_int = pars[3]
    alpha = pars[-1]
    R = opts[0]
    dl = opts[1]
    
    delta = R/dl
    
    gamma = lambda s: (s/alpha*(1 + ka_int/(s/alpha+kd_int)))**0.5
    phi = lambda s: alpha * delta * gamma(s) * mp.tanh(gamma(s))
    f = lambda s: ka * (s+phi(s))/(s + kd + phi(s))
#     Fp = 1 / mp.cosh((f+s)**0.5)
    
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)
 

    fp = lambda s: 1 / mp.cosh((f(s)+s)**0.5)

    if inversion == "ILap":
        Fl = map( lambda x: mp.invertlaplace(fp, x, method = 'talbot', dps = 5, degree = 5), t[1:])
        Flow = [0]
        Flow.extend(Fl)
#         for i in t[1:]:
#             Fl = mp.invertlaplace(fp, i, method='talbot')
#             Fl = sympify(Fl)
#             Flow.append(Fl)
        Flow = np.array(sympify(list(Flow)))
    elif inversion == 'ifft':
        
        Flow = doifft(Tmax=T, N = Nt+1, F = fp)[:Nt+1]
        
    Flow[Flow<0] = 0    
    numpy_array = np.array((t, np.asarray(Flow)), dtype=float)

    return numpy_array

def pulse(pars, opts, model, T=1, dt=0.001, Nx=1000, **kwargs):
    if model == 'rev_ads':
        array = analytical_adsdes(pars, opts, T, dt, Nx)
    elif model == 'analytical_7P':
        array = analytical_7P(pars, opts, T, dt, Nx, **kwargs)
    elif model == 'analytical_EqMP':
        array = analytical_EqMP(pars, opts, T, dt, Nx, **kwargs)
    elif model == 'analytical_DAMP':
        array = analytical_DAMP(pars, opts, T, dt, Nx, **kwargs)
    elif model == 'analytical_DAMPint':
        array = analytical_DAMPint(pars, opts, T, dt, Nx, **kwargs)
    return array

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

def moments(pulse_array, order):
    x_data, y_data = pulse_array[0], pulse_array[1]
    dt = x_data[1]-x_data[0]
    integrand = y_data * x_data ** order
    return np.trapz(integrand, x=x_data, dx=dt, axis=0)

def pulse_stand_metric(pulse_array):
    time, flow = pulse_array[0], pulse_array[1]
    peaktime = time[np.argmax(flow)]
    m0 = moments(pulse_array, 0)
    m1 = moments(pulse_array, 1)
    m2 = moments(pulse_array, 2)
    mean_res_time = m1/m0
    mom_dispers = m2 / m0 - mean_res_time**2
    return mean_res_time, peaktime, mom_dispers

def sens_pars(pars, opts, model, T=1, dt=0.001, Nx=1000, **kwargs):
    flow = [100]
    n = 0

    while flow[-1] > 0.001:
        array = pulse(pars = pars, opts = opts, model = model, T=T, dt=dt, Nx=Nx, **kwargs)
        time, flow = array[0], array[1]
        mean_flow = flow.mean()
        dispersion = sum((flow-mean_flow)**2)
        m = 0
        while dispersion < 1e-3:
            T = 2*T
            dt = 2*dt
            m += 1
            array = pulse(pars = pars, opts = opts, model = model, T=T, dt=dt, Nx=Nx, **kwargs)
            time, flow = array[0], array[1]
            mean_flow = flow.mean()
            dispersion = sum((flow-mean_flow)**2)
            if m > 5:
                break
        T = 2*T
        dt = 2*dt
        n += 1
        if n >20:
            break
    return pulse_stand_metric(array)    

def pulse_to_dim(array, opts):
    Temp = opts['T']                                    
    Temp_ref = opts['Tref']                              
    D_ref = opts['Dref zones']
    Length = opts['length']                                                              
    Area = opts['cross_section']                     
    eps = opts['porosity']                                        
    Mref = opts['Mref']                             
    M = opts['M']                                  
    Np = opts['Np']                                 

    D = D_ref[-1]*(Temp*Mref/M/Temp_ref)**0.5
    L = np.sum(Length)
    dim_flow = array[1]*D*Np/eps/L**2
    t = array[0]*eps*L**2/D
    return np.array([t, dim_flow])

def pulse_to_dim_less(array, pars, opts):
    Temp = opts['T']                                    
    Temp_ref = opts['Tref']                              
    D_ref = opts['Dref zones']
    Length = opts['length']                                                              
    Area = opts['cross_section']                     
    eps = opts['porosity']                                        
    Mref = opts['Mref']                             
    M = opts['M']                                  
    Np = opts['Np']                                 

    D = D_ref[-1]*(Temp*Mref/M/Temp_ref)**0.5
    L = np.sum(Length)
    dim_less_flow = array[1] / D / Np*eps*L**2
    tau = array[0]/eps/L**2*D
    return np.array([tau, dim_less_flow])

def pars_to_dim_less(pars, opts, model):
    Temp = opts['T']                                   
    Temp_ref = opts['Tref']                              
    Length = opts['length']                                                              
    Area = opts['cross_section']                    
    eps = opts['porosity']                          
    eps_p = opts['microporosity']
    D_ref = opts['Dref zones']                     
    Mref = opts['Mref']                             
    M = opts['M']                                    

    Np = opts['Np']                                

    R = opts['micropore_Rp']                        
    dr = opts['dr']                                  
    Sw = opts['Surface per weight']                        
    AS = opts['AS concentration']
    ASpore = opts['AS concentration within particle']
    mass = opts['Sample mass']
    Ac = AS / Sw 
    Sv = Sw * mass / Area / (Length[1]) / (1-eps)
    
#     assert D_ref[0] == 0 and D_ref[2] == 0 and Length[0] == 0 and Length[2] == 0, 'The model is for 1 Zone, be sure to correct D_ref and Length for 1 Zone'
    
    D = D_ref[1]*(Temp*Mref/M/Temp_ref)**0.5
    alpha = (eps * Length[1]**2/ D) * (pars[-1]/eps_p/R**2)
    if model == 'analytical_EqMP':
        pars_dless = [pars[0], alpha]
    elif model == 'analytical_DAMP':
        ka_dless = pars[0] * Ac*Sv*(1-eps)*Length[1]**2 / D
        kd_dless = pars[1] * eps*Length[1]**2 / D
        pars_dless = [ka_dless, kd_dless, alpha]
    elif model == 'analytical_DAMPint':
        ka_dless = pars[0] * Ac*Sv*(1-eps)*Length[1]**2 / D
        kd_dless = pars[1] * eps*Length[1]**2 / D
        ka_int_dless = pars[2] * ASpore * R**2/pars[-1]
        kd_int_dless = pars[3] * eps_p*R**2/pars[-1]
        pars_dless = [ka_dless, kd_dless, ka_int_dless, kd_int_dless, alpha]
    elif model == 'analytical_7P':
        ka_dless = pars[0] * Ac*Sv*(1-eps)*Length[1]**2 / D
        kd_dless = pars[1] * eps*Length[1]**2 / D
        k_ent_dless = pars[2] * eps*Length[1]**2 / D
        k_ext_dless = pars[3] * eps*Length[1]**2 / D
        ka_int_dless = pars[4] * ASpore * R**2/pars[-1]
        kd_int_dless = pars[5] * eps_p*R**2/pars[-1]
        pars_dless = [ka_dless, kd_dless, k_ent_dless, k_ext_dless, ka_int_dless, kd_int_dless, alpha]
    else:
        return 'Unknown model'
    return pars_dless

def pars_to_dim(pars, opts, model):
    Temp = opts['T']                                   
    Temp_ref = opts['Tref']                              
    Length = opts['length']                               
    dx = opts['dx']                                  
    Area = opts['cross_section']                    
    eps = opts['porosity']                          
    eps_p = opts['microporosity']
    D_ref = opts['Dref zones']                     
    Mref = opts['Mref']                             
    M = opts['M']                                    

    Np = opts['Np']                                

    R = opts['micropore_Rp']                        
    dr = opts['dr']                                  
    Sw = opts['Surface per weight']                        
    AS = opts['AS concentration']
    ASpore = opts['AS concentration within particle']
    mass = opts['Sample mass']
    Ac = AS / Sw 
    Sv = Sw * mass / Area / (Length[1]) / (1-eps)
    
#     assert D_ref[0] == 0 and D_ref[2] == 0 and Length[0] == 0 and Length[2] == 0, 'The model is for 1 Zone, be sure to correct D_ref and Length for 1 Zone'
    
    D = D_ref[1]*(Temp*Mref/M/Temp_ref)**0.5
#     alpha = (eps * Length[1]**2/ D) * (pars[-1]/eps_p/R**2)
    Dp = pars[-1] * eps_p*R**2/(eps * Length[1]**2/ D)
    if model == 'analytical_EqMP':
        pars_dim = [pars[0], Dp]
    elif model == 'analytical_DAMP':
        ka_dim = pars[0] / Ac / Sv / (1-eps) / Length[1]**2 * D
        kd_dim = pars[1] / eps / Length[1]**2 * D
        pars_dim = [ka_dim, kd_dim, Dp]
    elif model == ' analytical_7P':
        ka_dim = pars[0] / Ac / Sv / (1-eps) / Length[1]**2 * D
        kd_dim = pars[1] / eps / Length[1]**2 * D
        k_ent_dim = pars[2] / eps / Length[1]**2 * D
        k_ext_dim = pars[3] / eps / Length[1]**2 * D
        ka_int_dim = pars[4] / ASpore / R**2 * Dp
        kd_int_dim = pars[5] / eps_p / R**2 * Dp
        pars_dim = [ka_dim, kd_dim, k_ent_dim, k_ext_dim, ka_int_dim, kd_int_dim, Dp]
    else:
        return 'Unknown model'
    return pars_dim
        
def multiresponse(TD_values, temperatures, opts, model, dimensionless = True, **kwargs):
    dH_ads = TD_values['dH_ads']
    dS_ads = TD_values['dS_ads']

    Dp0 = TD_values['Dp0']
    Ediff = TD_values['Ediff']
    dH_ads_int = TD_values['dH_ads_int']
    dS_ads_int = TD_values['dS_ads_int']
    E_b = TD_values['E_bond']
    E_ent = TD_values['E_ent']
    E_ext = TD_values['E_ext']
    k_ent0 = TD_values['k_ent0']
    k_ext0 = TD_values['k_ext0']
    time = 100
    dt = 1

    Sv = opts['Surface per weight'] * opts['Sample mass'] / opts['cross_section'] / (opts['length'][1]) / (1-opts['porosity'])

    multipulse = []
    
    if model == 'analytical_7P':
        for temperature in temperatures:        
            K_ads = np.exp(-(dH_ads-temperature*dS_ads)/8.314/temperature)
            ka = ka_hertz_knud(opts = opts, temp = temperature)
            kd = ka / K_ads
            
#             kd = k_des_frenkel("C4H10", temperature)
#             ka = K_ads*kd
            
            K_ads_int = np.exp(-(dH_ads_int-temperature*dS_ads_int)/8.314/temperature)
            ka_int = ka_hertz_knud(opts = opts, temp = temperature)
            kd_int = ka_int / K_ads_int
            
#             kd_int = kd
#             ka_int = K_ads_int * kd_int
            
            Dp = Dp0 * np.exp(-Ediff/8.314/temperature)
            k_ent = k_ent0 * np.exp(-E_ent/8.314/temperature)
            k_ext = k_ext0 * np.exp(-E_ext/8.314/temperature)
            pars = [ka, kd, k_ent, k_ext, ka_int, kd_int, Dp]
            opts['T'] = temperature
            pars_dimless = pars_to_dim_less(pars, opts, model)
            opt_dimless = [opts['micropore_Rp'] * Sv]
            array = analytical_7P(pars = pars_dimless, opts = opt_dimless, T = time, dt = dt, Nx=1000, **kwargs)
            if dimensionless is False:
                array = pulse_to_dim(array, opts)
            multipulse.append(array)
    elif model == 'analytical_EqMP':
        for temperature in temperatures:
            K_ads = np.exp(-(dH_ads-temperature*dS_ads)/8.314/temperature)
            Dp = Dp0 * np.exp(-Ediff/8.314/temperature)
            pars = [K_ads, Dp]
            opts['T'] = temperature
            pars_dimless = pars_to_dim_less(pars, opts, model)
            opt_dimless = [opts['micropore_Rp'] * Sv, opts['porosity'], opts['microporosity']]
            array = analytical_EqMP(pars = pars_dimless, opts = opt_dimless, T = time, dt = dt, Nx=1000, **kwargs)
            if dimensionless is False:
                array = pulse_to_dim(array, opts)
            multipulse.append(array)
    elif model == 'analytical_DAMP':       
        for temperature in temperatures:
            K_ads = np.exp(-(dH_ads-temperature*dS_ads)/8.314/temperature)
            
            ka = ka_hertz_knud(opts = opts, temp = temperature)
            kd = ka / K_ads
            
#             kd = k_des_frenkel_manual("C4H10", temperature, E_b)
#             ka = K_ads*kd
            Dp = Dp0 * np.exp(-Ediff/8.314/temperature)
            pars = [ka, kd, Dp]
            opts['T'] = temperature
            pars_dimless = pars_to_dim_less(pars, opts, model)
            opt_dimless = [opts['micropore_Rp']/5.5, opts['unit_layer_thickness']]
            array = analytical_DAMP(pars = pars_dimless, opts = opt_dimless, T = time, dt = dt, Nx=1000, **kwargs)
            if dimensionless is False:
                array = pulse_to_dim(array, opts)
            multipulse.append(array)
    elif model == 'analytical_DAMPint':
        for temperature in temperatures:
            K_ads = np.exp(-(dH_ads-temperature*dS_ads)/8.314/temperature)

#             kd = k_des_frenkel_manual("C4H10", temperature, E_b)
#             ka = K_ads*kd
 
            ka = ka_hertz_knud(opts = opts, temp = temperature)
            kd = ka / K_ads

            K_ads_int = np.exp(-(dH_ads_int-temperature*dS_ads_int)/8.314/temperature)
    
#             kd_int = k_des_frenkel("C3H6", temperature)
#             ka_int = K_ads_int * kd_int
            
            ka_int = ka_hertz_knud(opts = opts, temp = temperature)
            kd_int = ka_int / K_ads_int
            Dp = Dp0 * np.exp(-Ediff/8.314/temperature)
            pars = [ka, kd, ka_int, kd_int, Dp]
            opts['T'] = temperature
            pars_dimless = pars_to_dim_less(pars, opts, model)
            opt_dimless = [opts['micropore_Rp']/5.5, opts['unit_layer_thickness']]
            array = analytical_DAMPint(pars = pars_dimless, opts = opt_dimless, T = time, dt = dt, Nx=1000, **kwargs)
            if dimensionless is False:
                array = pulse_to_dim(array, opts)
            multipulse.append(array)
    return np.array(multipulse)

   