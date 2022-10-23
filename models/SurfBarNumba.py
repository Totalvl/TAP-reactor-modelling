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


@nb.njit(parallel = True)
def MicroPDiff(CpIn, Dp, dt, dr, Cs, k_ent, k_ext, opts):
    """
    Function to solve PDE of microporous diffusion for one spatial point x of reactor. Function supposed to be run in parallel
    Function is using scipy sparse solver
    Vectorized implementation of solver_BE_simple using also
    a sparse (tridiagonal) matrix for efficiency.
    """
    # Extracting parameters for solving PDE in micropores
    R = opts['micropore_Rp']
    Sv = opts['Surface per volume']
    Ac = opts['Active sites per surface']
    eps_p = opts['microporosity']
    Nr = int(round(R/dr))
    r = np.linspace(0, R, Nr+1)       # Mesh points in pore space

    # Make sure dr and dt are compatible with r and t

    dr = r[1] - r[0]
    F = Dp/eps_p * dt/dr**2    # Factor before matrix elements. Used to shorten formulas. 

    Nx = len(Cs)

    # Representation of sparse matrix and right-hand side
    diagonal = np.zeros(Nr+1)
    lower    = np.zeros(Nr)
    upper    = np.zeros(Nr)
    b2       = np.zeros((Nx, Nr+1))
    Cp   = np.zeros((Nx, Nr+1))      # solution array at t[n+1]
    Cp_n = np.zeros((Nx, Nr+1))      # solution at t[n]
    Flow = np.zeros(Nx)

    # Precompute sparse matrix
    diagonal[:] = 1 + 2*F
    lower[:] = -F  
    upper[:] = -F  

    diagonal[0] = 1 + 2*F
    upper[0] = -2*F
    diagonal[-1] = 1+2*F+2*F*dr*k_ext
    lower[-1] = -2*F

    Ap = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1], shape=(Nr+1, Nr+1),
        format='csr')
#     print(A2.todense())

    # Set initial condition
    Cp_n[:] = CpIn
  
    # boundary conditions
    b2[:] = Cp_n
    b2[:, -1] = b2[:, -1] + 2* dt * Ac * k_ent /eps_p / dr * Cs[:]  
    for i in nb.prange(Nx):
        Cp[i,:] = scipy.sparse.linalg.spsolve(Ap, b2[i])

    return Cp
   

@nb.njit(parallel = True)
def MicroPDiff_linalg(CpIn, Dp, Ap, dt, Nr, dr, Cs, F, k_ent, Ac, Sv):
   """
   Function to solve PDE of microporous diffusion for one spatial point x of reactor. Function supposed to be run in parallel
   Function is using numpy solver since it has better interaction with numba
   """

    Ny = len(Cs)

    b2       = np.zeros((Ny, Nr+1))
    Cp   = np.zeros((Ny, Nr+1))      # solution array at t[n+1]
    Cp_n = np.zeros((Ny, Nr+1))      # solution at t[n]
    Flow = np.zeros(Ny)

    Cp_n[:] = CpIn


    b2[:] = Cp_n
    b2[:, -1] = b2[:, -1] + 2 * F * dr * Ac * k_ent / Dp * Cs[:]  # boundary conditions
    for i in nb.prange(Ny):
        Cp[i,:] = np.linalg.solve(Ap, b2[i])

    return Cp



def solver(CI, dt, f, T, thetaI, CpI, opts, pars, er=0.000001,
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
    Sw = opts['Surface per weight']                  # Surface-to-volume ratio for the microporous particles, 1/m
    AS = opts['AS concentration']                    # Surface concentration of active sites mol/m2 
    mass = opts['Sample mass']                       # Sample mass, kg
    eps_p = opts['microporosity']                    # Porosity of zeolite particle itself                        
    D = D_ref*(Temp*Mref/M/Temp_ref)**0.5

    ka = pars[0]
    kd = pars[1]
    
    try:
        Dp = pars[-1]
        k_ent = pars[2]
        k_ext = pars[3]
    except: 
        Dp = 0
        k_ent = 0
        k_ext = 0

    
    import time
    t0 = time.time()
    
    L=np.array([0.0])

    for l in Length:
        L = np.append(L, l+L[-1]) 

    Nx=np.around(L/dx).astype(int)

    x=np.linspace(0,L[-1],Nx[-1]+1)   # mesh points in space
    
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time
    
    Nr = int(round(R/dr))
    r = np.linspace(0, R, Nr+1) # Mesh points in space
    dr = r[1]-r[0]
#     print(len(r))
    if pore_plot:
        pore_numb=int(input('Enter the number of pore concentration profile of which will be plotted'))
        if pore_numb < 1 or pore_numb >= (Nx[2]-Nx[1]):
            raise ValueError('Pore number cannot be 0 or more than total number of pores')

    u0   = zeros(Nx[-1]+1)   # solution array for gas concentration at t[n+1] first approximation k
    u1   = zeros(Nx[-1]+1)   # solution array for gas concentration at t[n+1] next approximation k+1
    u_n = zeros(Nx[-1]+1)   # solution at t[n] for gas concentration
    v0   = zeros(Nx[-1]+1)   # solution array for surface concentration at t[n+1] first approximation k 
    v1   = zeros(Nx[-1]+1)   # solution array for surface concentration at t[n+1] next approximation k+1
    v_n = zeros(Nx[-1]+1)   # solution at t[n] for surface concentration 
    vp0   = zeros(Nx[-1]+1)   # solution array at t[n+1]
    vp1   = zeros(Nx[-1]+1)   # solution array at t[n+1]
    Cp_n   = zeros(Nr+1)      # solution array for gas inside micropore concentration at t[n]
    arrCpn = np.empty([Nx[2]-Nx[1], len(r)]) # array for gas inside micropore concentration at t[n]
    arrCp0 = np.empty([Nx[2]-Nx[1], len(r)]) # solution array for gas inside micropore at t[n+1] first approximation k 
    arrCp1 = np.empty([Nx[2]-Nx[1], len(r)])  # solution array for gas inside micropore at t[n+1] next approximation k+1
    Cs     = np.zeros(Nx[2]-Nx[1]) # array of surface concentration that is used for setting up boundary conditions for solving micropore diffusion PDE

    """
    Basic formula in the scheme:
    0.5*(D[i+1] + D[i])*(u[i+1] - u[i]) -
    0.5*(D[i] + D[i-1])*(u[i] - u[i-1])
    0.5*(D[i+1] + D[i])*u[i+1]
    0.5*(D[i] + D[i-1])*u[i-1]
    -0.5*(D[i+1] + D*a[i] + D[i-1])*u[i]
    """
    # Coefficient for elements in diagonals in sparse matrix and column vectors
    alpha1 = 0.5 * dt / dx**2 / eps
    Ac = AS / Sw 
    Sv = Sw * mass / Area / (L[2]-L[1]) / (1-eps)
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
    
    # Since we need to recalculate matrix coefficient to reflect reversibility, let's define function that update and reassemble matrix for each iteration
    
    def update_matrix(v):
        diagonal[Nx[1]:Nx[2]]= 1 + b11*(1-v[Nx[1]:Nx[2]]) + alpha1*(D[Nx[1]+1:Nx[2]+1] + \
                                + 2*D[Nx[1]:Nx[2]] + D[Nx[1]-1:Nx[2]-1])
        A = scipy.sparse.diags(
            diagonals=[diagonal, lower, upper],
            offsets=[0, -1, 1],
            shape=(Nx[-1]+1, Nx[-1]+1),
            format='csr')
        return A
    
    # Since matrixes for solving transport PDEs are the same for each equation let's define general matrix that will be used for solving every PDE
    pore_diagonal = np.zeros(Nr+1)
    pore_lower    = np.zeros(Nr)
    pore_upper    = np.zeros(Nr)
    Ap = np.zeros((Nr+1, Nr+1))
    
    # Coefficient for assembling matrixes of microporous transport PDEs
    F = Dp/eps_p * dt/dr**2

    # Precompute sparse matrix of MP PDE
    pore_diagonal[:] = 1 + 2*F
    pore_lower[:] = -F  
    pore_upper[:] = -F  
    
    # Insert boundary conditions
    pore_diagonal[0] = 1 + 2*F
    pore_upper[0] = -2*F
    pore_diagonal[-1] = 1+2*F+2*dt*k_ext / dr / Sv
    pore_lower[-1] = -2*F

    Ap = np.diag(pore_diagonal) + np.diag(pore_lower, -1) + np.diag(pore_upper, 1)
    #print A.todense()

    # Set initial condition
    for i in range(0,Nx[-1]+1):
        u_n[i] = CI(x[i])
        v_n[i]=thetaI(x[i])
#     print(u_n, v_n)    
    for i in range(0, Nr+1):
        Cp_n[i] = CpI(r[i])
    arrCpn[:] = Cp_n

    if user_action is not None:
        user_action(u_n, x, t, 0)

    Flow = np.array([0.0])
    iterations = np.array([0])
    # Time loop
    for n in range(0, Nt):
               
        if plot_update:
            clear_output(wait=True)

        A = update_matrix(v_n)
        # Calculating right-hand side of equation
        b[1:Nx[1]] = u_n[1:Nx[1]] + dt*f(x[1:Nx[1]], t[n+1]) 
        b[Nx[1]:Nx[2]] = u_n[Nx[1]:Nx[2]] + b12*v_n[Nx[1]:Nx[2]]+ dt*f(x[Nx[1]:Nx[2]], t[n+1])
        b[Nx[2]:Nx[3]] = u_n[Nx[2]:Nx[3]] + dt*f(x[Nx[2]:Nx[3]], t[n+1])
        # Boundary conditions
        b[0]  = u_n[0]
        b[-1] = 0
        u0[:] = scipy.sparse.linalg.spsolve(A, b)
        t_solv_1eq = time.process_time()
        t3 = (t_solv_1eq - t_matr_upd)/ (10 ** 9)
        
        it1=0
        
        erv = 10000
        eru = 10000
        time_fl1 = time.process_time()
      
        # Starting loop of iterative calculation and refinement of bulk and surface concentration 
        while erv > er*np.max(v0) or eru > er*np.max(u0):
            arrCp0[:] = arrCpn 
            v0[Nx[1]:Nx[2]] = (v_n[Nx[1]:Nx[2]] + b21 * u0[Nx[1]:Nx[2]] + dt * k_ext * arrCp0[:,-1] *eps_p / Sv / Ac) /\
                            (1 + b22 + b21 * u0[Nx[1]:Nx[2]] + dt * k_ent)

            # To remove accidental negative surface concentration that may occur at close to reliability threshold set of parameters
            v0[v0 < 0] = 0
            v0[v0 > 1] = 1

           
            it2 = 0
            # If permeation rate is set 0 skip the whole looping part
            if k_ext != 0 and k_ent != 0 and Dp != 0:
                vp0[:] = v0
                erCp = 10000
                ervp  = 10000

                while ervp > er*np.max(v0) or erCp > er*np.max(arrCp0):                                
                    
                   # Solbing in parallel MP transport PDE for each spatial point of catalyst
                    arrCp0[:] = MicroPDiff_linalg(Cs = vp0[Nx[1]:Nx[2]], CpIn = arrCpn, Dp = Dp, Ap = Ap, dt = dt, Nr = Nr,
                                                   dr = dr, F=F, k_ent = k_ent, Ac = Ac, Sv = Sv)

                    vp1[Nx[1]:Nx[2]] = (b21 * u0[Nx[1]:Nx[2]] + v_n[Nx[1]:Nx[2]] + dt * k_ext * arrCp0[:,-1] *eps_p / Sv / Ac) /\
                            (1 + b22 + b21 * u0[Nx[1]:Nx[2]] + dt * k_ent)
                    arrCp1[:] = MicroPDiff_linalg(Cs = vp1[Nx[1]:Nx[2]], CpIn = arrCpn, Dp = Dp, Ap = Ap, dt = dt, Nr = Nr,
                                                   dr = dr, F=F, k_ent = k_ent, Ac = Ac, Sv = Sv)


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
            
            eru = np.max(abs(u1-u0))
            erv = np.max(abs(v1-v0))

            it1+=1
            u0[:] = u1

            
        # Recording the time of the end of the iteration part
        time_fl4 = time.process_time()
        tfl4 = time_fl4 - time_fl1
        if plot_update:
            print(it1+it2, t[n], tfl4, np.max(v1[Nx[1]:Nx[2]]))
        # Plotting concentration profile inside the pores for certain spatial point
        if pore_plot:
            plt.figure(figsize=(7,5)) 
            plt.plot(r,arrCp0[pore_numb-1], 'r-')
            plt.xlabel('Space coordinate')
            plt.ylabel('Intesity')
            plt.title('t=%g' % t[n])
            plt.show()
        # Plotting concentration of surface species profile
        if surface_plot:
            plt.figure(figsize=(7,5)) 
            plt.plot(x[Nx[1]:Nx[2]],v0[Nx[1]:Nx[2]], 'r-')
            plt.xlabel('Space coordinate')
            plt.ylabel('Intesity')
            plt.title('t=%g' % t[n])
            plt.show() 
        # Recording the number of iterations and adding a point to the flow array
        iterations = np.append(iterations, it1)
        Fa = Area*D[-1]*(-(u0[-1]-u0[-2])/dx)
        Flow = np.append(Flow, Fa)
        
        if user_action is not None:
            user_action(u0, x, t, n+1)
            
        # Switch variables before next step
        u_n[:], u0[:] = u0, u_n
        v_n[:] = v1
        arrCpn[:] = arrCp0
        
    # Creating Pandas DF just as an alternative for data storage
#     df=pd.DataFrame({'time': t, 'Flow': np.asarray(Flow), 'iterations per step': np.asarray(iterations)})
    numpy_array = np.array((t, np.asarray(Flow), np.asarray(iterations)))
    t1 = time.process_time()

    return t1-t0, numpy_array


def pulse(opts, pars, T=1, dt = 0.001, Nx=100,  save_plot=False, show_plot=False, **kwargs):

    
    global images
    images=[]
    
    
    
    plot_method=2
    # Compute dt from Nx and F
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
    # Defining  initial surface concentration function as funciton of space  
    def thetaI(x):
        return 0
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