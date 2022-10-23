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

import cupy
import cupyx
from cupyx.scipy.sparse.linalg import lsqr
import numba as nb
from numba import cuda


"""
Solve the diffusion equation
    u_t = (D(x)*u_x)_x + f(x,t)
on (0,L) with boundary conditions u(0,t)_x = 0 and u(L,t) = 0,
for t in (0,T]. Initial condition: u(x,0) = I(x) - delta function
The following naming convention of variables are used.
===== ==========================================================
Name  Description
===== ==========================================================
Nx       The total number of mesh cells of reactor space; mesh points are numbered
         from 0 to Nx.
T        The stop time for the simulation.
I        Initial condition (Python function of x).
a        Resolution ratio (constant).
L        Length of the reactor space domain ([0,L]).
x        Mesh points in space.
t        Mesh points in time.
n        Index counter in time.
u0       Unknown bulk concentration of gas at new time level at previous iteration.
u1       Unknown bulk concentration of gas at new time level at next iteration.
u_n      Bulk concentration at the previous time level.
Cp_n     Initial distribution of gas within the pore at t=0 (Python function of x)
Fl_n     Flow to/from the pore at t=0 in case of inhomogenous distribution of gas along the pore. 
Fl0      Flow to/from the pore at previous iteration
Fl1      Flow to/from the pore at next iteration
arrCpn   2D array containing gas distribtuion in each pore along the pore-length 
         at previous time-step. Each row is 1 pore. 
arrCp0   2D array containing gas distribtuion in each pore along the pore-length 
         at new time level at previous iteration. 
arrCp1   2D array containing gas distribtuion in each pore along the pore-length 
         at new time level at next iteration. 
dx       Constant mesh spacing in x.
dt       Constant mesh spacing in t.
===== ==========================================================
``user_action`` is a function of ``(u, x, t, n)``, ``u[i]`` is the
solution at spatial mesh point ``x[i]`` at time ``t[n]``, where the
calling code can add visualization, error computations, data analysis,
store solutions, etc.
"""

@nb.jit(parallel = True)
def MicroPDiff1(CpIn, Nr, Cs, F, Nb):
    """
    Vectorized implementation of Backward Euler scheme FOR 1 TIMESTEP using also
    a sparse (tridiagonal) matrix for efficiency. The function is used to short
    the text of iteration cycle 
    ===== ==========================================================
    Name  Description
    ===== ==========================================================
    Nr       Mesh resolution on the pore-length scale; mesh points are numbered
             from 0 to Nr.
    CpIn     Initial distribution of gas within the pore at t=0 (Python function of x).
    Dp       Diffusivity within the pore (constant).
    R        Length of the pore space domain ([0,R]).
    r        Mesh points in pore-length space.
    Cp       Unknown porous concentration at new time level.
    Cp_n     Porous concentration at the previous time level.
    dr       Constant mesh spacing in r.
    dt       Constant mesh spacing in t.
    ===== ==========================================================
    """
    # Factor before matrix elements. Used to shorten formulas. 
    Nx = Nb

    # Representation of sparse matrix and right-hand side
    diagonal = cupy.zeros(Nr+1)
    lower    = cupy.zeros(Nr)
    upper    = cupy.zeros(Nr)
    b2       = cupy.zeros((Nx, Nr+1))
    Cp   = cupy.zeros((Nx, Nr+1))      # solution array at t[n+1]

    # Precompute sparse matrix
    diagonal[:] = 1 + 2*F
    lower[:] = -F  #1
    upper[:] = -F  #1

    diagonal[0] = 1 + 2*F
    upper[0] = -2*F
    diagonal[-1] = 1
    lower[-1] = 0
    A = cupyx.scipy.sparse.diags(
    diagonals=[diagonal, lower, upper],
    offsets=[0, -1, 1],
    format = 'csr')

    b2[:] = CpIn
    # boundary conditions 
    b2[:,-1] = Cs[:]
    
    for i in nb.prange(Nx):
        Cp[i,:] = lsqr(A, b2[i])[0]
        # Flow[i]=Dp*(-(Cp[i,-1]-Cp[i,-2])/dr)
    
    return Cp.get()

def solver(CI, dt, f, T, CpI, opts, pars, er=0.000001,
           user_action=None, pore_plot=False, surface_plot=False):
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
    R = opts['micropore_Rp']                        # Characteristic length of micropore diffusion, m
    dr = opts['dr']                                 # Micropore-scale grid step, m
    Np = opts['Np']                                  # Pulse size, mol 
    mass = opts['Sample mass']                       # Sample mass, kg
    Sw = opts['Outer surface per weight']            # Surface-to-weight ratio for the microporous particles, m2/kg
    Sv = Sw*mass/Area/Length[1]/(1-eps[1])           # Surface-to-volume ratio for the microporous particles, m2/m3
    a_s = opts['AS concentration active'] / Sw       # Surface concentration of active sites mol/m2                    
    a_s_tot = opts['AS concentration total']         # Total surface concentration of active sites mol/m2 
    a_s_int = (a_s_tot - opts['AS concentration active'])\
    *mass/Area/Length[1]/(1-eps[1])                  #Internal surface concentration of active sites mol/m2      
                      
    D = D_ref*(Temp*Mref/M/Temp_ref)**0.5

    KH = pars[0]
    Dp = pars[1]
    
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
    print(len(r))

    if pore_plot:
        pore_numb=int(input('Enter the number of pore concentration profile of which will be plotted'))
        if pore_numb < 1 or pore_numb >= (Nx[2]-Nx[1]):
            raise ValueError('Pore number cannot be 0 or more than total number of pores')


    u0     = zeros(Nx[-1]+1)   # solution array at t[n+1] at previous iteration
    u1     = zeros(Nx[-1]+1)   # solution array at t[n+1] at new iteration
    u2     = zeros(Nx[-1]+1)
    u_n    = zeros(Nx[-1]+1)   # solution at t[n]
    Cp_n   = zeros(Nr+1)
    Fl_n   = np.zeros(Nx[2]-Nx[1])
    Fl0    = np.zeros(Nx[2]-Nx[1])
    Fl1    = np.zeros(Nx[2]-Nx[1])
    arrCpn = np.empty([Nx[2]-Nx[1], len(r)])
    arrCp0 = np.empty([Nx[2]-Nx[1], len(r)])
    arrCp1 = np.empty([Nx[2]-Nx[1], len(r)])
    Cs     = np.zeros(Nx[2]-Nx[1]) 


    """
    Basic formula in the scheme:
    0.5*(D[i+1] + D[i])*(u[i+1] - u[i]) -
    0.5*(D[i] + D[i-1])*(u[i] - u[i-1])
    0.5*(D[i+1] + D[i])*u[i+1]
    0.5*(D[i] + D[i-1])*u[i-1]
    -0.5*(D[i+1] + D*a[i] + D[i-1])*u[i]
    """

    alpha1=0.5*dt/dx**2/eps

    # Representation of sparse matrix and right-hand side
    diagonal = zeros(Nx[-1]+1)
    lower    = zeros(Nx[-1])
    upper    = zeros(Nx[-1])
    b      = zeros(Nx[-1]+1)
    
    # Precompute sparse matrix (scipy format)

    diagonal[1:-1] = 1 + alpha1*(D[2:] + 2*D[1:-1] + D[:-2])
    lower[:-1] = -alpha1*(D[1:-1] + D[:-2])    
    upper[1:]  = -alpha1*(D[2:] + D[1:-1])
    
    # Insert boundary conditions
    diagonal[0] = 1 + 2*alpha1*(D[1] + D[0])
    upper[0] = -2*alpha1*(D[1] + D[0])
    diagonal[-1] = 1
    lower[-1] = 0

    A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1],
        shape=(Nx[-1]+1, Nx[-1]+1),
        format='csr')
    #print A.todense()

    # Set initial condition
    for i in range(0,Nx[-1]+1):
        u_n[i]  = CI(x[i])
    
    for i in range(0, Nr+1):
        Cp_n[i] = CpI(r[i])
    arrCpn[:] = Cp_n
    Fl_n[:] = Dp*(-(Cp_n[-1]-Cp_n[-2])/dr-(Cp_n[-2]-Cp_n[-3])/dr)/2
    

    if user_action is not None:
        user_action(u_n, x, t, 0)
    Flow=[]
    iterations=[]
    iterations.append(0)
    Flow.append(0)
    # Time loop
    
    for n in range(0, Nt):
        # print(t[n])
        #u_n[u_n<0]=0
        b[1:-1] = u_n[1:-1] + dt*f(x[1:-1], t[n]) 
        # Boundary conditions
        b[0]  = u_n[0]
        b[-1] = 0
        
        # Condition to check if there is adsorption/diffusion involved  in process. If not, proceed as regular bulk diffusion calculation. 
        if Dp !=0 and KH!=0:
            b[Nx[1]:Nx[2]] = u_n[Nx[1]:Nx[2]]+ Fl_n[:] * Sw * mass / Area / (L[2]-L[1]) / eps * dt + dt*f(x[Nx[1]:Nx[2]], t[n])
        
        u0[:] = scipy.sparse.linalg.spsolve(A, b)
        # Setting up initial value of errors so the first loop for rectifying concentration values occur
        it1  = 0
        erCp = 10000
        eru  = 10000
        erFl = 10000
        
        if Dp != 0 and KH != 0 :
            time_fl1 = time.perf_counter()
            # Start iterating to calculate pore concentration and bulk concentration 
            while eru > er*np.max(u0) or erFl > er:

                F = Dp/eps_p * dt/dr**2
                Nb = len(u0[Nx[1]:Nx[2]])
                # Getting first approximation array of concentration distribution in micropores for every spatial point in catalyst zone
                arrCp0[:] = MicroPDiff1(Cs = cupy.asarray(KH*u0[Nx[1]:Nx[2]]), CpIn = cupy.asarray(arrCpn), F = F, 
                                                   Nr = Nr, Nb = Nb)
                Fl0[:]=Dp*(-(arrCp0[:,-1]-arrCp0[:,-2])/dr)


                b[Nx[1]:Nx[2]] = u_n[Nx[1]:Nx[2]]+ Fl0[:] * Sw * mass / Area / (L[2]-L[1]) / eps * dt + dt*f(x[Nx[1]:Nx[2]], t[n])

                u1[:] = scipy.sparse.linalg.spsolve(A, b)
                
                # Getting first approximation array of concentration distribution in micropores for every spatial point in catalyst zone
                arrCp1[:] = MicroPDiff1(Cs = cupy.asarray(KH*u0[Nx[1]:Nx[2]]), CpIn = cupy.asarray(arrCpn), F = F, 
                                                   Nr = Nr, Nb = Nb)
                Fl1[:]=Dp*(-(arrCp1[:,-1]-arrCp1[:,-2])/dr)

                b[Nx[1]:Nx[2]] = u_n[Nx[1]:Nx[2]]+ Fl1[:] * Sw * mass / Area / (L[2]-L[1]) / eps * dt + dt*f(x[Nx[1]:Nx[2]], t[n])
                u2[:] = scipy.sparse.linalg.spsolve(A, b)
                eru = np.max(abs(u1-u0))
                erFl = np.max(abs(Fl1-Fl0))
                # Setting up initial guess for the next iteration
                Fl0[:] = Fl1
                u0[:] = u2
                it1+=1

                

            time_fl4 = time.perf_counter()
            tfl4 = time_fl4 - time_fl1
            print(it1, t[n], tfl4)
            # Plotting concentration profile inside the pores for certain spatial point
            if pore_plot:
                plt.plot(r,arrCp0[pore_numb-1], 'r-')
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
        arrCpn[:] = arrCp0
        Fl_n[:]=Fl0
        
        
    # df=pd.DataFrame({'time': t, 'Flow': np.asarray(Flow), 'iterations per step': np.asarray(iterations)}) - code line in case if you want to store data as pandas
    numpy_array = np.array((t, np.asarray(Flow), np.asarray(iterations)))
    t1 = time.perf_counter()

    return t1-t0, numpy_array


def pulse(opts, pars, T=1, dt = 0.001, Nx=100,  save_plot=False, show_plot=False, **kwargs):

    
    global images
    images=[]
    
    
    
    plot_method=2
    Length = opts['length']    
    L = [0]
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
    # Defining  initial Microporous gas concentration function as funciton of pore-space 
    def CpI(r):
        return 0
    # Defining free member of PDE (function for surplus or conversion)     
    def f(x,t):
        return 0
    
    if save_plot:
        for name in glob.glob('./frames/tmp_*.png'):
            os.remove(name)
        for name in glob.glob('./Gifs/*.gif'):
            os.remove(name)
    # Plotting function        
    def plot_u(u, x, t, n):
        global images
        if t[n] == 0:
            time.sleep(2)
        if plot_method > 0:
            plt.plot(x,u, 'r-')
            plt.axvline(x=opts['length'][1], color='y', linestyle='--')
            plt.axvline(x=opts['length'][2], color='y', linestyle='--')
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

    print(t, 's')
    return array 