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
u     Bulk conventration at current/new time level.
u_n   u at the previous time level.
v     Surface concentration at current/new time level
v_n   v at the previos time level
dx    Constant mesh spacing in x.
dt    Constant mesh spacing in t.
===== ==========================================================
``user_action`` is a function of ``(u, x, t, n)``, ``u[i]`` is the
solution at spatial mesh point ``x[i]`` at time ``t[n]``, where the
calling code can add visualization, error computations, data analysis,
store solutions, etc.
"""

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
    


def solver_lin(CI, dt, f, T, thetaI, opts, pars, er=0.000001,
           user_action=None):
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

    Temp = opts['T']                                 # Temperature, K
    Temp_ref = opts['Tref']                          # Reference temperature for Knudsen diffusivity, K
    
    Length = opts['length']                          # List of zone lengths, m 
    dx = opts['dx']                                  # Reactor-scale grid step, m
    Area = opts['cross_section']                     # Reactor cross-section, m^2
    eps = opts['porosity']                           # List of zone porosities
    
    D_ref = array(opts['D_ref'])                     # Reference Knudsen diffusivity, m^2/s 
    Mref = opts['Mref']                              # Reference molecular mass for Knudsen diffusivity, g/mol
    M = opts['M']                                    # Molecular mass of the diffusing gas, g/mol

    Np = opts['Np']                                  # Pulse size, mol 
    mass = opts['Sample mass']                       # Sample mass, kg
    Sw = opts['Outer surface per weight']            # Surface-to-weight ratio for the microporous particles, m2/kg
    Sv = Sw*mass/Area/Length[1]/(1-eps[1])           # Surface-to-volume ratio for the microporous particles, m2/m3
    a_s = opts['AS concentration active'] / Sw       # Surface concentration of active sites mol/m2                    
    a_s_tot = opts['AS concentration total']         # Total surface concentration of active sites mol/m2 
    a_s_int = (a_s_tot - opts['AS concentration active'])\
    *mass/Area/Length[1]/(1-eps[1])                  #Internal surface concentration of active sites mol/m2 
    D = D_ref*(Temp*Mref/M/Temp_ref)**0.5            # Calculating Diffusivity at given temperature for given gas
    ka = pars[0]                                     # Defining adsorption parameters
    kd = pars[1]
    
    # Setting up time 0 to count time of solution
    import time
    t0 = time.process_time()
    
    # Quick cycle to create array with coordinates of zones' beginning 
    L=[0]
    for l in Length:
        L.append(l+L[-1])
    L = np.array(L)
    
    Nx=np.around(L/dx).astype(int)
    x=np.linspace(0,L[-1],Nx[-1]+1)   # mesh points in space
    
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time

    if isinstance(D, (float,int)):
        D = zeros(Nx+1) + D


    u0   = zeros(Nx[-1]+1)   # solution array for gas concentration at t[n+1] first approximation k
    u1   = zeros(Nx[-1]+1)   # solution array for gas concentration at t[n+1] next approximation k+1
    u_n = zeros(Nx[-1]+1)   # solution at t[n] for gas concentration
    v0   = zeros(Nx[-1]+1)   # solution array for surface concentration at t[n+1] first approximation k 
    v1   = zeros(Nx[-1]+1)   # solution array for surface concentration at t[n+1] next approximation k+1
    v_n = zeros(Nx[-1]+1)   # solution at t[n] for surface concentration 

    """
    Basic formula in the scheme:
    0.5*(D[i+1] + D[i])*(u[i+1] - u[i]) -
    0.5*(D[i] + D[i-1])*(u[i] - u[i-1])
    0.5*(D[i+1] + D[i])*u[i+1]
    0.5*(D[i] + D[i-1])*u[i-1]
    -0.5*(D[i+1] + D*a[i] + D[i-1])*u[i]
    """

    alpha1=0.5*dt/dx**2/eps # Coefficient for elements in diagonals in sparse matrix and column vectors
    b11=AS*mass/Area/(L[2]-L[1])/eps*dt*ka
    b12=AS*mass/Area/(L[2]-L[1])/eps*dt*kd
    b21=ka*dt 
    b22=kd*dt

    # Representation of sparse matrix and right-hand side
    diagonal = zeros(Nx[-1]+1)
    lower    = zeros(Nx[-1])
    upper    = zeros(Nx[-1])
    b        = zeros(Nx[-1]+1)

    # Precompute sparse matrix (scipy format)
    diagonal[1:Nx[1]] = 1 + alpha1*(D[2:Nx[1]+1] + 2*D[1:Nx[1]] + D[:Nx[1]-1])
    diagonal[Nx[1]:Nx[2]]= 1 + b11 + alpha1*(D[Nx[1]+1:Nx[2]+1] + 2*D[Nx[1]:Nx[2]] + D[Nx[1]-1:Nx[2]-1])
    diagonal[Nx[2]:Nx[3]]= 1 + alpha1*(D[Nx[2]+1:Nx[3]+1] + 2*D[Nx[2]:Nx[3]] + D[Nx[2]-1:Nx[3]-1])
    
    lower[:-1] = -alpha1*(D[1:-1] + D[:-2])    
    upper[1:]  = -alpha1*(D[2:] + D[1:-1])
    
    # Insert boundary conditions
    diagonal[0] = 1 + 2*alpha1*(D[1] + D[0])
    upper[0] = -2*alpha1*(D[1] + D[0])
    diagonal[-1] = 1
    lower[-1] = 0
    
    # Assemble matrix
    A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1],
        shape=(Nx[-1]+1, Nx[-1]+1),
        format='csr')

    # Set initial condition
    for i in range(0,Nx[-1]+1):
        u_n[i] = CI(x[i])
        v_n[i]=thetaI(x[i])
#     print(u_n, v_n)    
    
    # point of plotting or extracting current concentration
    if user_action is not None:
        user_action(u_n, x, t, 0)
    Flow=[]
    iterations=[]
    iterations.append(0)
    Flow.append(0)
    # Time loop
    for n in range(0, Nt): 
        # Calculating column vector using initial conditions
        b[1:Nx[1]] = u_n[1:Nx[1]] + dt*f(x[1:Nx[1]], t[n+1])
        b[Nx[1]:Nx[2]] = u_n[Nx[1]:Nx[2]] + b12*v_n[Nx[1]:Nx[2]]+ dt*f(x[Nx[1]:Nx[2]], t[n+1])
        b[Nx[2]:Nx[3]] = u_n[Nx[2]:Nx[3]] + dt*f(x[Nx[2]:Nx[3]], t[n+1])
        # Boundary conditions
        b[0]  = u_n[0]
        b[-1] = 0
        u0[:] = scipy.sparse.linalg.spsolve(A, b)
        
        # Setting ip first iteration
        it=0
        
        # Setting up first error estimation
        erv = 10000 
        eru = 10000
        
        # Starting loop of iterative calculation and refinement of bulk and surface concentration 
        while erv > er*np.max(v0) or eru > er*np.max(u0):
            v0[Nx[1]:Nx[2]] = (b21*u0[Nx[1]:Nx[2]]+v_n[Nx[1]:Nx[2]])/(1+b22)

            v0[v0 < 0] = 0
            v0[v0 > 1] = 1
            
            b[1:Nx[1]] = u_n[1:Nx[1]] + dt*f(x[1:Nx[1]], t[n+1]) 
            b[Nx[1]:Nx[2]] = u_n[Nx[1]:Nx[2]] + b12*v0[Nx[1]:Nx[2]]+ dt*f(x[Nx[1]:Nx[2]], t[n+1])
            b[Nx[2]:Nx[3]] = u_n[Nx[2]:Nx[3]] + dt*f(x[Nx[2]:Nx[3]], t[n+1])
            
            u1[:] = scipy.sparse.linalg.spsolve(A, b)
            
            v1[Nx[1]:Nx[2]] = (b21*u1[Nx[1]:Nx[2]]+v_n[Nx[1]:Nx[2]])/(1+b22)
            v1[v1 < 0] = 0
            v1[v1 > 1] = 1

            eru = np.max(abs(u1-u0))
            erv = np.max(abs(v1-v0))
            it+=1

            u0[:] = u1

#         print('iterations number', it)
        iterations.append(it)
        Fa=Area*D[-1]*(-(u0[-1]-u0[-2])/dx)
        Flow.append(Fa)
        
        if user_action is not None:
            user_action(u0, x, t, n+1)
            
        # Switch variables before next step
        u_n[:] = u0
        v_n[:] = 

    # Assembling solution array
    numpy_array = np.array((t, np.asarray(Flow), np.asarray(iterations)))
    t1 = time.process_time()

    return t1-t0, numpy_array
  
def pulse(opts, pars, T=1, dt = 0.001, Nx=100,  save_plot=False, show_plot=False):
    global images
    images=[]
    
    
    
    plot_method=2
    # Compute dt from Nx and F
    Length = opts['length']
    L=[0]
    for l in Length:
        L.append(l+L[-1])
    L = np.array(L)
    opts['dx'] = L[-1]/Nx
    eps=opts['porosity']
    A = opts['cross_section']
    dx = opts['dx']
    Np = opts['Np']
    
    # Defining  initial bulk concentration function as funciton of space
    def CI(x):
        I0=Np/A/eps*2/dx/(np.pi)**(1/2)/special.erf(1/dx)
        return I0*np.exp(-(x/dx)**2)
   # Defining  initial surface concentration function as funciton of space 
   def thetaI(x):
        return 0
    # Defining free member of PDE (function for surplus or conversion)    
    def f(x,t):
        return 0
    
    # Configuring plotting parameters
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
    
    
    
    # Function for creating a domain of Diffusivities as function of space D(x). Needed for heterogenous (multi-zone) systems
    def fill_a(Dref, L, dx):
        Nzone = len(Dref)
        assert len(Dref) == len(L)-1
        a = np.zeros(int(round(L[-1]/float(dx)))+1)
        Nx=np.around(L/dx).astype(int)
        for i in range(Nzone):
            a[Nx[i]:Nx[i+1]+1] = Dref[i]

        return a
    
    # Filling the domain of diffusivities for each space point
    D=fill_a(opts['Dref zones'],L,dx)
    opts['D_ref'] = D         
    
    # Running the solver and plotting
    if show_plot:
        user_action = plot_u
    else:
        user_action = None
    t, array =solver(CI, dt, f, T, thetaI, CpI, opts=opts, pars=pars, user_action=user_action, **kwargs)
    print(np.argmax(array[1]))
    peaktime = array[0][np.argmax(array[1])]
    lbl="Peak time " + str(peaktime)
    plt.plot(array[0],array[1], 'r-', markersize=4, label=lbl)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Flow')
    plt.show() 
    
    if save_plot:
        imageio.mimsave('./Gifs/movie.gif', images)    #saving GIF image

#     print(t, 's')
    return array