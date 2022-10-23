import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sympy import sympify
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

def analytical_eqMPdif(pars, opts, T=1, dt=0.001, Nx=1000):
    

    Temp = opts['T']                                    # Temperature, K
    Temp_ref = opts['Tref']                             # Reference temperature for Knudsen diffusivity, K
  
    L = opts['length'][-1]                               # List of zone lengths, m 
    eps = opts['porosity']                           # List of zone porosities

    KH = pars[0]
    Dp = pars[1]
    D_ref = opts['Dref zones'][0]                     # Reference Knudsen diffusivity, m^2/s
    Mref = opts['Mref']                              # Reference molecular mass for Knudsen diffusivity, g/mol
    M = opts['M']                                    # Molecular mass of the diffusing gas, g/mol

    Np = opts['Np']                                  # Pulse size, mol
    Sw = opts['Surface per weight']                        # Surface-to-volume ratio for the microporous particles, 1/m
    mass = opts['Sample mass'] 
    
    D = D_ref*(Temp*Mref/M/Temp_ref)**0.5

    eps=opts['porosity']
    Area = opts['cross_section']
    eps_p = opts['Mircoporous porosity within particle']
    R = opts['micropore_Rp']

    KH = pars[0]
    Dp = pars[1]
    dx = L/Nx
    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)
    x = np.linspace(0, L, Nx+1)
    Sv = Sw*mass/Area/L/(1-eps)
    tau = t * D / eps / L**2
    C=0
    Fl=0
    n=10000   

    tb = eps * L**2 / D
    tp = eps_p * R**2 / Dp
    alpha = tb / tp
    t = np.linspace(0, 1, 1001) 
    tl = t/tb

    fp = lambda s: Area*D/L*Np/(eps*Area*L)/mp.cosh((((1-eps)*L**2*Sv/D)*(KH/(1+KH))*\
                                               Dp/R*(s/alpha)**(1/2)*mp.tanh((s/alpha)**(1/2))+s)**(1/2))

    Flow = []
    Flow.append(0)
    for i in tl[1:]:
        Fl = mp.invertlaplace(fp, i, method='talbot')
        Fl = sympify(Fl)
        Flow.append(Fl)
    # Flow = np.array(Flow)

    df=pd.DataFrame({'time': t, 'Flow': np.asarray(Flow)})

    return df

def pulse(pars, opts, model, T=1, dt=0.001, Nx=1000):
    if model == 'rev_ads':
        datf = analytical_adsdes(pars, opts,T, dt, Nx)
    elif model == 'pore_eq_lin':
        datf = analytical_eqMPdif(pars, opts, T, dt, Nx)
    return datf