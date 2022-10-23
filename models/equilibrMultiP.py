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
import multiprocessing as mp
from functools import partial
from itertools import product

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


def MicroPDiff(num, Cs, CpIn, Dp, R, dt, dr, eps_p, Cs2 = None):
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
    Nr = int(round(R/dr))
    r = np.linspace(0, R, Nr+1)       # Mesh points in pore space
    
    # Make sure dr and dt are compatible with r and t
        
    dr = r[1] - r[0]
    F = Dp/eps_p * dt/dr**2    # Factor before matrix elements. Used to shorten formulas. 
    Cp   = np.zeros(Nr+1)      # solution array at t[n+1]
    Cp_n = np.zeros(Nr+1)      # solution at t[n]
    
    # Representation of sparse matrix and right-hand side
    diagonal = np.zeros(Nr+1)
    lower    = np.zeros(Nr)
    upper    = np.zeros(Nr)
    b2       = np.zeros(Nr+1)

    # Precompute sparse matrix
    diagonal[:] = 1 + 2*F
    lower[:] = -F  #1
    upper[:] = -F  #1
    
    # Insert boundary conditions
    diagonal[0] = 1 + 2*F
    upper[0] = -2*F
    
    diagonal[Nr] = 1
    lower[-1] = 0

    A2 = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1], shape=(Nr+1, Nr+1),
        format='csr')
#     print(A2.todense())

    # Set initial condition
    Cp_n[:] = CpIn[num, :]


    b2[:] = Cp_n
    # boundary conditions
    b2[-1] = Cs[num]  
    Cp[:] = scipy.sparse.linalg.spsolve(A2, b2)
    print(Cp)

    # Estimating derivative at the pore outlet point to calculate flow.
    # try:
    #    Flow=Dp*(-(Cp[-1]-Cp[-2])/dr-(Cp[-2]-Cp[-3])/dr)/2
    # except:
    #    Flow=Dp*(-(Cp[-1]-Cp[-2])/dr)
    return Cp




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
#     internal_BC_type = opts['internal_BC_type']      # '1' - boundary as a CSTR; '2' - flux continuity
#     micropore_geometry = opts['micropore_geometry']  # 'linear' or 'spherical'
#     micropore_BC_type = opts['micropore_BC_type']    # 'kinetics' or 'equilibrium'

    Temp = opts['T']                                    # Temperature, K
    Temp_ref = opts['Tref']                              # Reference temperature for Knudsen diffusivity, K
    
    Length = opts['length']                               # List of zone lengths, m 
    dx = opts['dx']                                  # Reactor-scale grid step, m
    Area = opts['cross_section']                     # Reactor cross-section, m^2
    eps = opts['porosity']                           # List of zone porosities
    
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
    KH = pars[0]
    Dp = pars[1]
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
    print(len(r))
    
    
    if isinstance(D, (float,int)):
        D = zeros(Nx+1) + D
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

    

#     # Precompute sparse matrix (scipy format)
#     diagonal[1:-1] = 1 + alpha1*(D[2:] + 2*D[1:-1] + D[:-2])
#     lower[:-1] = -alpha1*(D[1:-1] + D[:-2])    
#     upper[1:]  = -alpha1*(D[2:] + D[1:-1])
    
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
    
    pool = mp.Pool(mp.cpu_count())
    
    for n in range(0, Nt):
        # print(t[n])
        
        b[1:-1] = u_n[1:-1] + dt*f(x[1:-1], t[n]) 
        # Boundary conditions
        b[0]  = u_n[0]
        b[-1] = 0
        

        if Dp !=0 and KH!=0:
#             for j, i in enumerate(u_n[Nx[1]:Nx[2]]):
#                 arrCp0[j], Fl0[j] = MicroPDiff(CpIn = arrCpn[j], Dp = Dp, R = R, dt = dt, 
#                                                dr=dr, Cs = KH*i/(1+KH), eps_p = eps_p)
            b[Nx[1]:Nx[2]] = u_n[Nx[1]:Nx[2]]+ Fl_n[:] * Sw * mass / Area / (L[2]-L[1]) / eps * dt + dt*f(x[Nx[1]:Nx[2]], t[n])
        u0[:] = scipy.sparse.linalg.spsolve(A, b)

        it1  = 0
        erCp = 10000
        eru  = 10000
        erFl = 10000
        
        if Dp != 0 and KH != 0 :
            time_fl1 = time.perf_counter()
            while eru > er*np.max(u0):
                
                ArrErrCp = []
                
#                 print(np.min(abs(KH*u0[Nx[1]:Nx[2]])/(1+KH)))
                
                
                wrapMPD=partial(MicroPDiff, Cs = KH/(1+KH)*u0[Nx[1]:Nx[2]], CpIn = arrCpn,  Dp = Dp, R = R, dt = dt, 
                                                   dr=dr, eps_p = eps_p)
            
                i = np.linspace(0, Nx[2]-Nx[1]-1, Nx[2]-Nx[1]).astype(int)
                
                arrCp0 = pool.map(wrapMPD, i)
                
                arrCp0=np.array(arrCp0)

                Fl0[:] = Dp*(-(arrCp0[:,-1]-arrCp0[:,-2])/dr)

                    
                
                b[Nx[1]:Nx[2]] = u_n[Nx[1]:Nx[2]]+ Fl0[:] * Sw * mass / Area / (L[2]-L[1]) / eps * dt + dt*f(x[Nx[1]:Nx[2]], t[n])
                
                u1[:] = scipy.sparse.linalg.spsolve(A, b)

                eru = np.max(abs(u1-u0))
#                 erFl = np.max(abs(Fl1-Fl0))

#                 Fl0[:] = Fl1
#                 print(eru)
#                 arrCp0[:] = arrCp1
                u0[:] = u1
                it1+=1
#            time_fl2 = time.perf_counter()
#            time_flow_res = time_fl2 - time_fl1

                
            time_fl4 = time.perf_counter()
            tfl4 = time_fl4 - time_fl1
            print(it1, t[n], tfl4)
            if pore_plot:
                plt.plot(r,arrCp0[pore_numb-1], 'r-')
                plt.xlabel('Space coordinate')
                plt.ylabel('Intesity')
                plt.title('t=%g' % t[n])
                plt.show()
            if surface_plot:
                plt.plot(x[Nx[1]:Nx[2]],u0[Nx[1]:Nx[2]]*KH/(1+KH), 'r-')
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
        
        
    df=pd.DataFrame({'time': t, 'Flow': np.asarray(Flow), 'iterations per step': np.asarray(iterations)})

    t1 = time.perf_counter()

    return t1-t0, df


def pulse(opts, pars, T=1, dt = 0.001, Nx=100,  save_plot=False, show_plot=False, **kwargs):

    
    global images
    images=[]
    
    
    
    plot_method=2
    # Compute dt from Nx and F
    L = opts['length']
    opts['dx'] = L[-1]/Nx
    eps=opts['porosity']
    A = opts['cross_section']
    dx = opts['dx']
    Np = opts['Np']
    

    def CI(x):
        I0=Np/A/eps*2/dx/(np.pi)**(1/2)/special.erf(1/dx)
        return I0*np.exp(-(x/dx)**2)
    def CpI(r):
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
        t, datf=solver(CI, dt, f, T, CpI, opts=opts, pars=pars, user_action=plot_u, **kwargs)
#         (I, a, f, L, Nx, D, T, theta=0.5, u_R=0, user_action=plot_u)
    else:
        t, datf=solver(CI, dt, f, T, CpI, opts=opts, pars=pars, user_action=None, **kwargs)
    
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
#     print(max(datf['Flow']))
    print(t, 's')
    return datf    