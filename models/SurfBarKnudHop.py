import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
from numpy import linspace, zeros, random, array
import numpy as np
import time, sys
import pandas as pd 
import glob, os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import imageio
import io
from PIL import Image
from scipy import special
import numba as nb
from IPython.display import clear_output


"""
Solve the diffusion equation
    u_t = (a(x)*u_x)_x + f(x,t)
on (0,L) with boundary conditions u(0,t) = u_L and u(L,t) = u_R,
for t in (0,T]. Initial condition: u(x,0) = I(x).
The following naming convention of variables are used.
===== ==========================================================
Name  Description
===== ==========================================================
Nx    The total number of mesh cells; mesh points are numbered
      from 0 to Nx.
T     The stop time for the simulation.
I     Initial condition (Python function of x).
a     Variable coefficient (constant).
L     Length of the domain ([0,L]).
x     Mesh points in space.
t     Mesh points in time.
n     Index counter in time.
u     Unknown at current/new time level.
u_n   u at the previous time level.
dx    Constant mesh spacing in x.
dt    Constant mesh spacing in t.
===== ==========================================================
``user_action`` is a function of ``(u, x, t, n)``, ``u[i]`` is the
solution at spatial mesh point ``x[i]`` at time ``t[n]``, where the
calling code can add visualization, error computations, data analysis,
store solutions, etc.
"""
# @nb.jit(parallel = True)
def MicroPDiff(CpIn, thetaIntIn, dt, dr, Cs, opts, pars_int):
    """
    Vectorized implementation of solver_BE_simple using also
    a sparse (tridiagonal) matrix for efficiency.
    """
    R = opts['micropore_Rp']
    Sv = opts['Surface per volume']
    Ac = opts['Active sites per surface']
    eps_p = opts['microporosity']
    ASpore = opts['AS concentration within particle']
    err = 0.000001
    
    assert len(pars_int) == 5
    
    k_ent = pars_int[0]
    k_ext = pars_int[1]
    ka_int = pars_int[2]
    kd_int = pars_int[3]
    Dp = pars_int[4]
    
    
    Nr = int(round(R/dr))
    r = np.linspace(0, R, Nr+1)       # Mesh points in pore space

    # Make sure dr and dt are compatible with r and t

    dr = r[1] - r[0]
    beta = Dp/eps_p * dt/dr**2    # Factor before matrix elements. Used to shorten formulas. 
    k1 = dt/eps_p * ka_int * ASpore
    k2 = dt/eps_p * kd_int * ASpore
    Nx = len(Cs)

    # Representation of sparse matrix and right-hand side
    diags = np.zeros((Nx,Nr+1))
    lwr = np.zeros((Nx, Nr))
    upr = np.zeros((Nx, Nr))
    b2 = np.zeros((Nx, Nr+1))
    Cp_1 = np.zeros((Nx, Nr+1))      # solution array at t[n+1]
    Cp_2 = np.zeros((Nx, Nr+1))
    Cp_n = np.zeros((Nx, Nr+1))      # solution at t[n]
    Flow = np.zeros(Nx)

    # Precompute sparse matrix
    lwr[:,:] = - beta  #1
    upr[:,:] = - beta  #1

    upr[:, 0] = -2 * beta
    lwr[:, -1] = -2 * beta
    
#     print(A2.todense())

    # Set initial condition
#     for i in range(0,Nr+1):
    Cp_n[:] = CpIn

    thetaInt_1 = thetaIntIn.copy() 
    thetaInt_2 = thetaIntIn.copy() 
    
    
    itjn = np.zeros(Nx)
    
    for i in range(Nx):
        erCp = 10000
        erTheta = 10000
        it = 0
        while erCp > err*np.max(Cp_n) or erTheta > err*np.max(thetaIntIn):
            b2[i,:] = Cp_n[i] + k2 * thetaInt_1[i]
            b2[i, -1] = b2[i, -1] + 2* dt * Ac * k_ent /eps_p / dr * Cs[i]  # boundary conditions
            diags[i,:] = 1 + 2*beta + k1*(1-thetaInt_1[i])
            diags[i,-1] = 1 + 2*beta + k1*(1-thetaInt_1[i,-1]) + 2*dt*k_ext / dr / Sv

            Cp_1[i,:] = scipy.sparse.linalg.spsolve(scipy.sparse.diags(
                                        diagonals=[diags[i], lwr[i], upr[i]], 
                                        offsets=[0, -1, 1], shape=(Nr+1, Nr+1),
                                        format='csr'),
                                        b2[i])
            # Flow[i]=Dp*(-(Cp[i,-1]-Cp[i,-2])/dr)
            thetaInt_1[i, :] = (thetaIntIn[i, :] + dt*ka_int*Cp_1[i,:])/(1+dt*kd_int+dt*ka_int*Cp_1[i,:])
            b2[i, :] = Cp_n[i] + k2 * thetaInt_1[i]
            b2[i, -1] = b2[i, -1] + 2* dt * Ac * k_ent /eps_p / dr * Cs[i]  # boundary conditions
            diags[i,:] = 1 + 2*beta + k1*(1-thetaInt_1[i])
            diags[i,-1] = 1 + 2*beta + k1*(1-thetaInt_1[i, -1]) + 2*dt*k_ext / dr / Sv

            Cp_2[i,:] = scipy.sparse.linalg.spsolve(scipy.sparse.diags(
                                        diagonals=[diags[i], lwr[i], upr[i]], 
                                        offsets=[0, -1, 1], shape=(Nr+1, Nr+1),
                                        format='csr'),
                                        b2[i])
                
            thetaInt_2[i, :] = (dt*ka_int*Cp_2[i,:] + thetaIntIn[i, :])/(1+dt*kd_int+dt*ka_int*Cp_2[i,:]) 

            erCp = np.max(abs(Cp_2[i,:]-Cp_1[i,:]))
            
            erTheta = np.max(abs(thetaInt_2[i, :] - thetaInt_1[i, :]))

            thetaInt_1[i,:] = thetaInt_2[i,:]
            Cp_1[i,:] = Cp_2[i,:]
            it+=1
            itjn[i] = it
            
    # Update u_n before next step
    #u_n[:] = u
#     Cp_n[:], Cp[:] = Cp, Cp_n
#     print(itjn.max())
    return Cp_1[:], thetaInt_1


# @nb.jit()
def MicroPDiff_linalg(CpIn, thetaIntIn, dt, dr, Cs, opts, pars_int):
    """
    Vectorized implementation of solver_BE_simple using also
    a sparse (tridiagonal) matrix for efficiency.
    
    
    """
    R = opts['micropore_Rp']
    Sv = opts['Surface per volume']
    Ac = opts['Active sites per surface']
    eps_p = opts['microporosity']
    ASpore = opts['AS concentration within particle']
    err = 0.000001
    assert len(pars_int) == 5
    
    k_ent = pars_int[0]
    k_ext = pars_int[1]
    ka_int = pars_int[2]
    kd_int = pars_int[3]
    Dp = pars_int[4]
    
    Nr = int(round(R/dr))
    r = np.linspace(0, R, Nr+1)       # Mesh points in pore space

    # Make sure dr and dt are compatible with r and t

    dr = r[1] - r[0]
    beta = Dp/eps_p * dt/dr**2    # Factor before matrix elements. Used to shorten formulas. 
    k1 = dt/eps_p * ka_int * ASpore
    k2 = dt/eps_p * kd_int * ASpore
    Nx = len(Cs)

    # Representation of sparse matrix and right-hand side
    diags = np.zeros((Nx,Nr+1))
    lwr = np.zeros((Nx, Nr))
    upr = np.zeros((Nx, Nr))
    b2 = np.zeros((Nx, Nr+1))
    Cp_1 = np.zeros((Nx, Nr+1))      # solution array at t[n+1]
    Cp_2 = np.zeros((Nx, Nr+1))
    Cp_n = np.zeros((Nx, Nr+1))      # solution at t[n]
    Ap = np.zeros((Nx, Nr+1, Nr+1))
    Flow = np.zeros(Nx)

    # Precompute sparse matrix
    lwr[:,:] = - beta  #1
    upr[:,:] = - beta  #1

    upr[:, 0] = -2 * beta
    lwr[:, -1] = -2 * beta
    
#     print(A2.todense())

    # Set initial condition
#     for i in range(0,Nr+1):
    Cp_n[:] = CpIn

    thetaInt_1 = thetaIntIn.copy() 
    thetaInt_2 = thetaIntIn.copy() 
    
    
    itjn = np.zeros(Nx)
    
    for i in range(Nx):
        erCp = 10000
        erTheta = 10000
        it = 0
        while erCp > err*np.max(Cp_n) or erTheta > err*np.max(thetaIntIn):
            b2[i,:] = Cp_n[i] + k2 * thetaInt_1[i]
            b2[i, -1] = b2[i, -1] + 2* dt * Ac * k_ent /eps_p / dr * Cs[i]  # boundary conditions
            diags[i,:] = 1 + 2*beta + k1*(1-thetaInt_1[i])
            diags[i, -1] = 1 + 2*beta + k1*(1-thetaInt_1[i,-1]) + 2*dt*k_ext / dr / Sv
                                                                    
            
#             for j in nb.prange(Nr+1):
#                 if j > 0 and j < Nr:
#                     Ap[i,j,j] = 1 + 2*beta + k1*(1-thetaInt_1[i, j])
#                     Ap[i,j,j+1] = - beta
#                     Ap[i,j,j-1] = - beta
#             Ap[i, 0, 0] = 1 + 2*beta + k1*(1-thetaInt_1[i, 0]) 
#             Ap[i, 0, 1] = - 2*beta 
#             Ap[i,-1,-1] = 1 + 2*beta + k1*(1-thetaInt_1[i, -1]) + 2*beta*dr*k_ext
#             Ap[i,-1,-2] = - 2*beta
            
            Ap[i] = np.diagflat(diags[i])+np.diagflat(upr[i],1)+np.diagflat(lwr[i],-1)
            Cp_1[i,:] = np.linalg.solve(Ap[i], b2[i])
            
            # Flow[i]=Dp*(-(Cp[i,-1]-Cp[i,-2])/dr)
            thetaInt_1[i, :] = (dt*ka_int*Cp_1[i,:] + thetaIntIn[i, :])/(1+dt*kd_int+dt*ka_int*Cp_1[i,:])
            b2[i, :] = Cp_n[i] + k2 * thetaInt_1[i]
            b2[i, -1] = b2[i, -1] + 2* dt * Ac * k_ent /eps_p / dr * Cs[i]  # boundary conditions
            diags[i,:] = 1 + 2*beta + k1*(1-thetaInt_1[i])
            diags[i, -1] = 1 + 2*beta + k1*(1-thetaInt_1[i, -1]) + 2*dt*k_ext / dr / Sv 
                                                                    
#             for j in nb.prange(Nr+1):
#                 if j > 0 and j < Nr:
#                     Ap[i,j,j] = 1 + 2*beta + k1*(1-thetaInt_1[i, j])
#                     Ap[i,j,j+1] = - beta
#                     Ap[i,j,j-1] = - beta
#             Ap[i, 0, 0] = 1 + 2*beta + k1*(1-thetaInt_1[i, 0]) 
#             Ap[i, 0, 1] = - 2*beta 
#             Ap[i,-1,-1] = 1 + 2*beta + k1*(1-thetaInt_1[i, -1]) + 2*beta*dr*k_ext
#             Ap[i,-1,-2] = - 2*beta
            
            Ap[i] = np.diagflat(diags[i])+np.diagflat(upr[i],1)+np.diagflat(lwr[i],-1)
            
            Cp_2[i,:] = np.linalg.solve(Ap[i], b2[i])
                
            thetaInt_2[i, :] = (dt*ka_int*Cp_2[i,:] + thetaIntIn[i, :])/(1+dt*kd_int+dt*ka_int*Cp_2[i,:]) 

            erCp = np.max(np.abs(Cp_2[i,:]-Cp_1[i,:]))
            erTheta = np.max(np.abs(thetaInt_2[i, :] - thetaInt_1[i, :]))
            thetaInt_1[i,:] = thetaInt_2[i, :]
            Cp_1[i,:] = Cp_2[i,:]
            it+=1
            itjn[i] = it
            
    # Update u_n before next step
    #u_n[:] = u
#     Cp_n[:], Cp[:] = Cp, Cp_n
#     print(itjn.max())
    return Cp_1[:], thetaInt_1


import scipy.sparse
import scipy.sparse.linalg
from numpy import linspace, zeros, random, array
import numpy as np
import time, sys
import pandas as pd 
import glob, os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import imageio
import io
from PIL import Image
from scipy import special
    


def solver(CI, dt, f, T, thetaI, thetaIntI, CpI, opts, pars, er=0.000001,
           user_action=None, pore_plot=False, surface_plot=False, plot_update=False):
    """
    The a variable is an array of length Nx+1 holding the values of
    a(x) at the mesh points.
    Method: (implicit) theta-rule in time.
    Nx is the total number of mesh cells; mesh points are numbered
    from 0 to Nx.
    D = dt/dx**2 and implicitly specifies the time step.
    T is the stop time for the simulation.
    I is a function of x.
    user_action is a function of (u, x, t, n) where the calling code
    can add visualization, error computations, data analysis,
    store solutions, etc.
    """

        # Options
#     internal_BC_type = opts['internal_BC_type']      # '1' - boundary as a CSTR; '2' - flux continuity
#     micropore_geometry = opts['micropore_geometry']  # 'linear' or 'spherical'
#     micropore_BC_type = opts['micropore_BC_type']    # 'kinetics' or 'equilibrium'

    Temp = opts['T']                                    # Temperature, K
    Temp_ref = opts['Tref']                              # Reference temperature for Knudsen diffusivity, K
    
    Length = opts['length']                               # List of zone lengths, m 
    dx = opts['dx']                                  # Reactor-scale grid step, m
    Area = opts['cross_section']                     # Reactor cross-section, m^2
    eps = opts['porosity']                           # List of zone porosities
    eps_p = opts['microporosity']
    D_ref = array(opts['D_ref'])                     # Reference Knudsen diffusivity, m^2/s 
    Mref = opts['Mref']                              # Reference molecular mass for Knudsen diffusivity, g/mol
    M = opts['M']                                    # Molecular mass of the diffusing gas, g/mol

    Np = opts['Np']                                  # Pulse size, mol
#     tau_p = opts['pTime']                            # Pulse peak time (Gaussian), s

    R = opts['micropore_Rp']                        # Characteristic length of micropore diffusion, m
    dr = opts['dr']                                  # Micropore-scale grid step, m
#     if micropore_BC_type=='equilibrium':
    Sw = opts['Surface per weight']                        # Surface-to-volume ratio for the microporous particles, 1/m
    AS = opts['AS concentration']
    ASpore = opts['AS concentration within particle']
    mass = opts['Sample mass']
#     if micropore_BC_type=='kinetics':
#         Ns = opts['micropore_Ns']                        # Total concentration of adsorption sites in the pore-mouth control volume, mol/m_s^3
#     D = opts['Diffusivities']                          
    D = D_ref*(Temp*Mref/M/Temp_ref)**0.5
    
#     print(min(D), max(D))
#     ri = lambda i: Rp - i*dr 
    # Parameters (for now, only isothermal)
#     Dp = pars[0]
#     if micropore_BC_type == 'kinetics':
    ka = pars[0]
    kd = pars[1]
    k_ent = pars[2]
    k_ext = pars[3]
    Dp = pars[-1]
    try:
        pars_int = np.array(pars[2:])
    except: 
        pars_int = np.array([0, 0, 0, 0, 0])
#     elif micropore_BC_type == 'equilibrium':
#         KH = pars[1]
#     else:
#         raise Exception('Unknown type of pore mouth boundary condition!')
    
    import time
    t0 = time.process_time()
    
    L=[0]
    for l in Length:
        L.append(l+L[-1])
    L = np.array(L)  
    
    Nx=np.around(L/dx).astype(int)
    x=np.linspace(0,L[-1],Nx[-1]+1)   # mesh points in space
    
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time
    
    Nr = int(round(R/dr))
    r = np.linspace(0, R, Nr+1) # Mesh points in space
#     print(len(r))
    if pore_plot:
        pore_numb=int(input('Enter the number of pore concentration profile of which will be plotted'))
        if pore_numb < 1 or pore_numb >= (Nx[2]-Nx[1]):
            raise ValueError('Pore number cannot be 0 or more than total number of pores')

    u0   = zeros(Nx[-1]+1)   # solution array at t[n+1]
    u1   = zeros(Nx[-1]+1)   # solution array at t[n+1]
    u_n = zeros(Nx[-1]+1)   # solution at t[n]
    v0   = zeros(Nx[-1]+1)   # solution array at t[n+1]
    v1   = zeros(Nx[-1]+1)   # solution array at t[n+1]
    v_n = zeros(Nx[-1]+1)   # solution at t[n]
    vp0   = zeros(Nx[-1]+1)   # solution array at t[n+1]
    vp1   = zeros(Nx[-1]+1)   # solution array at t[n+1]
    Cp_n   = zeros(Nr+1)
    thetaIntSP = zeros(Nr+1)
    Fl_n   = np.zeros(Nx[2]-Nx[1])
    Fl0    = np.zeros(Nx[2]-Nx[1])
    Fl1    = np.zeros(Nx[2]-Nx[1])
    arrCpn = np.empty([Nx[2]-Nx[1], len(r)])
    arrCp0 = np.empty([Nx[2]-Nx[1], len(r)])
    arrCp1 = np.empty([Nx[2]-Nx[1], len(r)])
    thetaInt_n = np.empty([Nx[2]-Nx[1], len(r)])
    thetaInt0 = np.empty([Nx[2]-Nx[1], len(r)])
    thetaInt1 = np.empty([Nx[2]-Nx[1], len(r)])
    Cs     = np.zeros(Nx[2]-Nx[1])

    """
    Basic formula in the scheme:
    0.5*(D[i+1] + D[i])*(u[i+1] - u[i]) -
    0.5*(D[i] + D[i-1])*(u[i] - u[i-1])
    0.5*(D[i+1] + D[i])*u[i+1]
    0.5*(D[i] + D[i-1])*u[i-1]
    -0.5*(D[i+1] + D*a[i] + D[i-1])*u[i]
    """

    alpha1 = 0.5 * dt / dx**2 / eps
    Ac = AS / Sw 
    Sv = Sw * mass / Area / (L[2]-L[1]) / (1-eps)
    opts['Surface per volume'] = Sv
    opts['Active sites per surface'] = Ac
    
    b11 = Ac * Sv * (1-eps) / eps * dt * ka
    b12 = Ac * Sv * (1-eps) / eps * dt * kd
    b21 = ka * dt
    b22 = kd * dt

    # Representation of sparse matrix and right-hand side
    diagonal = zeros(Nx[-1]+1)
    lower    = zeros(Nx[-1])
    upper    = zeros(Nx[-1])
    b        = zeros(Nx[-1]+1)

    # Precompute sparse matrix (scipy format)
    diagonal[1:Nx[1]] = 1 + alpha1*(D[2:Nx[1]+1] + 2*D[1:Nx[1]] + D[:Nx[1]-1])
    diagonal[Nx[2]:Nx[3]]= 1 + alpha1*(D[Nx[2]+1:Nx[3]+1] + 2*D[Nx[2]:Nx[3]] + D[Nx[2]-1:Nx[3]-1])
    
    lower[:-1] = -alpha1*(D[1:-1] + D[:-2])    
    upper[1:]  = -alpha1*(D[2:] + D[1:-1])
    
    # Insert boundary conditions
    diagonal[0] = 1 + 2*alpha1*(D[1] + D[0])
    upper[0] = -2*alpha1*(D[1] + D[0])
    diagonal[-1] = 1
    lower[-1] = 0

    def update_matrix(v):
        diagonal[Nx[1]:Nx[2]]= 1 + b11*(1-v[Nx[1]:Nx[2]]) + alpha1*(D[Nx[1]+1:Nx[2]+1] + \
                                + 2*D[Nx[1]:Nx[2]] + D[Nx[1]-1:Nx[2]-1])
        A = scipy.sparse.diags(
            diagonals=[diagonal, lower, upper],
            offsets=[0, -1, 1],
            shape=(Nx[-1]+1, Nx[-1]+1),
            format='csr')
        return A
        
    #print A.todense()

    # Set initial condition
    for i in range(0,Nx[-1]+1):
        u_n[i] = CI(x[i])
        v_n[i]=thetaI(x[i])
#     print(u_n, v_n)    
    for i in range(0, Nr+1):
        Cp_n[i] = CpI(r[i])
        thetaIntSP = thetaIntI(r[i])
    
    arrCpn[:] = Cp_n
    thetaInt_n[:] = thetaIntSP
    
    if user_action is not None:
        user_action(u_n, x, t, 0)
    Flow=[]
    iterations=[]
    iterations.append(0)
    Flow.append(0)
    # Time loop
    for n in range(0, Nt):
               
        if plot_update:
            clear_output(wait=True)

        A = update_matrix(v_n)

#         print('Matrix update time ', t2)
        
        b[1:Nx[1]] = u_n[1:Nx[1]] + dt*f(x[1:Nx[1]], t[n+1]) 
        b[Nx[1]:Nx[2]] = u_n[Nx[1]:Nx[2]] + b12*v_n[Nx[1]:Nx[2]]+ dt*f(x[Nx[1]:Nx[2]], t[n+1])
        b[Nx[2]:Nx[3]] = u_n[Nx[2]:Nx[3]] + dt*f(x[Nx[2]:Nx[3]], t[n+1])
        # Boundary conditions
        b[0]  = u_n[0]
        b[-1] = 0
        u0[:] = scipy.sparse.linalg.spsolve(A, b)

#         print('Solving 1 eqation ', t3)
        
        it1=0
        
        erv = 10000
        eru = 10000
        time_fl1 = time.time()
        while erv > er*np.max(v0) or eru > er*np.max(u0):
            arrCp0[:] = arrCpn 
            v0[Nx[1]:Nx[2]] = (b21 * u0[Nx[1]:Nx[2]] + v_n[Nx[1]:Nx[2]] + dt * k_ext * arrCp0[:,-1] *eps_p / Sv / Ac) /\
                            (1 + b22 + b21 * u0[Nx[1]:Nx[2]] + dt * k_ent)
#             print(max(v0))
            v0[v0 < 0] = 0
            v0[v0 > 1] = 1


            it2 = 0
            if k_ext != 0 and k_ent != 0 and Dp != 0:
                vp0[:] = v0
                erCp = 10000
                ervp  = 10000

                while ervp > er*np.max(v0) or erCp > er*np.max(arrCp0):                                
#                 print(np.min(abs(KH*u0[Nx[1]:Nx[2]])/(1+KH)))
                    arrCp0[:], thetaInt0[:] = MicroPDiff(Cs = vp0[Nx[1]:Nx[2]], CpIn = arrCpn, 
                                                         thetaIntIn = thetaInt_n, dt = dt, dr = dr, 
                                                         opts = opts, pars_int = pars_int)
                    
                    vp1[Nx[1]:Nx[2]] = (b21 * u0[Nx[1]:Nx[2]] + v_n[Nx[1]:Nx[2]] + dt * k_ext * arrCp0[:,-1] *eps_p / Sv / Ac) /\
                            (1 + b22 + b21 * u0[Nx[1]:Nx[2]] + dt * k_ent)
                    arrCp1[:], thetaInt1[:] = MicroPDiff(Cs = vp1[Nx[1]:Nx[2]], CpIn = arrCpn, 
                                                         thetaIntIn = thetaInt_n, dt = dt, dr=dr, 
                                                         opts = opts, pars_int = pars_int)


                    ervp = np.max(abs(vp1-vp0))
                    erCp = np.max(abs(arrCp1-arrCp0))
                    arrCp0[:] = arrCp1
                    vp0[:] = vp1
                    it2+=1

                v0[:] = vp0
#            time_fl2 = time.perf_counter()
#            time_flow_res = time_fl2 - time_fl1

            
            A = update_matrix(v0)
            
            b[1:Nx[1]] = u_n[1:Nx[1]] + dt*f(x[1:Nx[1]], t[n+1]) 
            b[Nx[1]:Nx[2]] = u_n[Nx[1]:Nx[2]] + b12*v0[Nx[1]:Nx[2]]+ dt*f(x[Nx[1]:Nx[2]], t[n+1])
            b[Nx[2]:Nx[3]] = u_n[Nx[2]:Nx[3]] + dt*f(x[Nx[2]:Nx[3]], t[n+1])
            
            u1[:] = scipy.sparse.linalg.spsolve(A, b)
            
            v1[Nx[1]:Nx[2]] = (b21 * u1[Nx[1]:Nx[2]] + v_n[Nx[1]:Nx[2]] + dt * k_ext * arrCp0[:,-1] *eps_p / Sv / Ac) /\
                            (1 + b22 + b21 * u1[Nx[1]:Nx[2]] + dt * k_ent)
            v1[v1 < 0] = 0
            v1[v1 > 1] = 1
#             print(np.max(v1[Nx[1]:Nx[2]]))
            
            eru = np.max(abs(u1-u0))
            erv = np.max(abs(v1-v0))
#             print(eru, erv)
            it1+=1
            u0[:] = u1
#             print(v0, v1)
#             print(u0, u1)
            
        
        time_fl4 = time.time()
        tfl4 = time_fl4 - time_fl1
        if plot_update:
            print(it1+it2, t[n], tfl4, np.max(v1[Nx[1]:Nx[2]]))
        if pore_plot:
            plt.figure(figsize=(7,5)) 
            plt.plot(r,arrCp0[pore_numb-1], 'r-')
            plt.xlabel('Space coordinate')
            plt.ylabel('Intesity')
            plt.title('t=%g' % t[n])
            plt.show()
        if surface_plot:
            plt.figure(figsize=(7,5)) 
            plt.plot(x[Nx[1]:Nx[2]],v0[Nx[1]:Nx[2]], 'r-')
            plt.xlabel('Space coordinate')
            plt.ylabel('Intesity')
            plt.title('t=%g' % t[n])
            plt.show() 

        iterations.append(it1)
        Fa = Area*D[-1]*(-(u0[-1]-u0[-2])/dx)
        Flow.append(Fa)
        
        if user_action is not None:
            user_action(u0, x, t, n+1)
            
        # Switch variables before next step
        u_n[:], u0[:] = u0, u_n
        v_n[:] = v1
        arrCpn[:] = arrCp0
        thetaInt_n[:] = thetaInt1
        
#     print(len(t), len(Flow), len(iterations))    
    df=pd.DataFrame({'time': t, 'Flow': np.asarray(Flow), 'iterations per step': np.asarray(iterations)})

    t1 = time.time()

    return t1-t0, df


def pulse(opts, pars, T=1, dt = 0.001, Nx=100,  save_plot=False, show_plot=False, **kwargs):

    
    global images
    images=[]
    
    
    
    plot_method=2
    # Compute dt from Nx and F
    L=[0]
    for l in opts['length']:
        L.append(l+L[-1])
    L = np.array(L)  
    opts['dx'] = L[-1]/Nx
    eps=opts['porosity']
    A = opts['cross_section']
    dx = opts['dx']
    Np = opts['Np']
    

    def CI(x):
        I0=Np/A/eps*2/dx/(np.pi)**(1/2)/special.erf(1/dx)
        return I0*np.exp(-(x/dx)**2)
    def thetaI(x):
        return 0
    def CpI(r):
        return 0
    def thetaIntI(r):
        return 0
    def f(x,t):
        return 0
    
    if save_plot:
        for name in glob.glob('./frames/tmp_*.png'):
            os.remove(name)
        for name in glob.glob('./Gifs/*.gif'):
            os.remove(name)
            
    def plot_u(u, x, t, n):
        global images
        if t[n] == 0:
            time.sleep(2)
        if plot_method > 0:
            plt.figure(figsize=(7,5)) 
            plt.plot(x,u, 'r-')
            plt.xlabel('Space coordinate')
            plt.ylabel('Intesity')
            plt.title('t=%g' % t[n])
            time.sleep(0.1) # pause between frames
            if save_plot:
                buf = io.BytesIO() #creating buffer to save image
                plt.savefig(buf, format='png', facecolor="w") #saving image
                buf.seek(0)  #accessing picture in the buffer
                images.append(imageio.imread(buf)) #appending image to GIF file
                buf.close() #cleaning the buffer
                plt.clf()
            if show_plot:
                plt.show()
    
    
    
    
    def fill_a(Dref, L, dx):
        Nzone = len(Dref)
        assert len(Dref) == len(L)-1
        a = np.zeros(int(round(L[-1]/float(dx)))+1)
        Nx=np.around(L/dx).astype(int)
        for i in range(Nzone):
            a[Nx[i]:Nx[i+1]+1] = Dref[i]

        return a
    
    
    D=fill_a(opts['Dref zones'],L,dx)
    opts['D_ref'] = D   
    if show_plot:
        t, datf=solver(CI, dt, f, T, thetaI, thetaIntI, CpI, opts=opts, pars=pars, user_action=plot_u, **kwargs)
#         (I, a, f, L, Nx, D, T, theta=0.5, u_R=0, user_action=plot_u)
    else:
        t, datf=solver(CI, dt, f, T, thetaI, thetaIntI, CpI, opts=opts, pars=pars, user_action=None, **kwargs)
    
    if save_plot:
        imageio.mimsave('./Gifs/movie.gif', images)    #saving GIF image
    if show_plot:    
        peaktime=datf.iloc[datf['Flow'].idxmax()]['time']
        lbl="Peak time " + str(peaktime)
        plt.plot(datf['time'],datf['Flow'], 'r-', markersize=4, label=lbl)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('Flow')
        plt.show()
    print(t, 's')
    return datf