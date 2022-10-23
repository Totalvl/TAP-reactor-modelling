from scipy.integrate import odeint
import scipy.odr as odr
from numpy import exp, zeros, arange, array
from tapper.experiment import *
from tapper.field import *
from tapper.io import *
import pandas as pd

pi = 3.14159
R = 8.314

def model_equations(t, y, pars, opts):

    dy = zeros(len(y))

    # Options
    internal_BC_type = opts['internal_BC_type']      # '1' - boundary as a CSTR; '2' - flux continuity
    micropore_geometry = opts['micropore_geometry']  # 'linear' or 'spherical'
    micropore_BC_type = opts['micropore_BC_type']    # 'kinetics' or 'equilibrium'

    T = opts['T']                                    # Temperature, K
    Tref = opts['Tref']                              # Reference temperature for Knudsen diffusivity, K
    
    L = opts['length']                               # List of zone lengths, m 
    dx = opts['dx']                                  # Reactor-scale grid step, m
    A = opts['cross_section']                        # Reactor cross-section, m^2
    eps = opts['porosity']                           # List of zone porosities
    
    D_ref = array(opts['D_ref'])                     # Reference Knudsen diffusivity, m^2/s 
    Mref = opts['Mref']                              # Reference molecular mass for Knudsen diffusivity, g/mol
    M = opts['M']                                    # Molecular mass of the diffusing gas, g/mol

    Np = opts['Np']                                  # Pulse size, mol
    tau_p = opts['pTime']                            # Pulse peak time (Gaussian), s

    Rp = opts['micropore_Rp']                        # Characteristic length of micropore diffusion, m
    dr = opts['dr']                                  # Micropore-scale grid step, m
    if micropore_BC_type=='equilibrium':
        Sv = opts['micropore_Sv']                        # Surface-to-volume ratio for the microporous particles, 1/m

    if micropore_BC_type=='kinetics':
        Ns = opts['micropore_Ns']                        # Total concentration of adsorption sites in the pore-mouth control volume, mol/m_s^3
                              
    D = D_ref*(T*Mref/M/Tref)**0.5
    ri = lambda i: Rp - i*dr 
    
    # Parameters (for now, only isothermal)
    Dp = pars[0]
    if micropore_BC_type == 'kinetics':
        ka = pars[1]
        kd = pars[2]
    elif micropore_BC_type == 'equilibrium':
        KH = pars[1]
    else:
        raise Exception('Unknown type of pore mouth boundary condition!')
    
    # Reactor-scale diffusion

    # Inlet boundary (to be reconstructed post-factum)
    start = int(0)
    for i in [start]:
        dy[i] = 0
        
    # The first internal point of the first zone
    start = int(1)
    for i in [start]:
        # Flux boundary condition - a one-parameter Gaussian pulse
        C_BC = ((2*Np*dx/D[0]/A)*(t*exp(-t/tau_p)/tau_p**2) + 4*y[i] - y[i+1])/3.0
        dy[i] = dy[i] + D[0]*(y[i+1] - 2*y[i] + C_BC)/(dx**2)/eps[0]
    
    # Inner first zone
    start = int(2)
    if internal_BC_type == '1':
        stop = int(L[0]/dx) - 1
    elif internal_BC_type == '2':
        stop = int(L[0]/dx) - 2
    else:
        raise Exception('Unknown type of internal BC!')
        
    for i in range(start, stop+1):
        dy[i] = dy[i] + D[0]*(y[i+1] - 2*y[i] + y[i-1])/(dx**2)/eps[0]

    if internal_BC_type == '2':
        idx = int(L[0]/dx) - 1
        C_BC = (4*D[0]*y[idx]-D[0]*y[idx-1]-D[1]*y[idx+3]+4*D[1]*y[idx+2])/(3*D[0] + 3*D[1])
        dy[idx] = dy[idx] + D[0]*(C_BC - 2*y[idx] + y[idx-1])/(dx**2)/eps[0]

    # 1-2 zone boundary
    start = int(L[0]/dx)
    for i in [start]:
        if internal_BC_type == '1':
            dy[i] = dy[i] + (2.0/dx/(eps[0]+eps[1]))*(D[1]*(-3*y[i]+4*y[i+1]-y[i+2]) - \
                                                      D[0]*(3*y[i]-4*y[i-1]+y[i-2]))/2.0/dx
        else:
            dy[i] = 0

    # Inner second zone

    if internal_BC_type == '1':
        start = int(L[0]/dx) + 1
    elif internal_BC_type == '2':
        idx = int(L[0]/dx) + 1
        C_BC = (4*D[1]*y[idx]-D[1]*y[idx+1]-D[0]*y[idx-3]+4*D[0]*y[idx-2])/(3*D[0] + 3*D[1])
        dy[idx] = dy[idx] + D[1]*(y[idx+1] - 2*y[idx] + C_BC)/(dx**2)/eps[1]
        
        start = int(L[0]/dx) + 2
    else:
        raise Exception('Unknown type of internal BC!')

    if internal_BC_type == '1':
        stop = int((L[0]+L[1])/dx) - 1
    elif internal_BC_type == '2':
        stop = int((L[0]+L[1])/dx) - 2
    else:
        raise Exception('Unknown type of internal BC!')
    
    for i in range(start, stop+1):
        dy[i] = dy[i] + D[1]*(y[i+1] - 2*y[i] + y[i-1])/(dx**2)/eps[1]

    if internal_BC_type == '2':
        idx = int((L[0]+L[1])/dx) - 1
        C_BC = (4*D[1]*y[idx]-D[1]*y[idx-1]-D[2]*y[idx+3]+4*D[2]*y[idx+2])/(3*D[1] + 3*D[2])
        dy[idx] = dy[idx] + D[1]*(C_BC - 2*y[idx] + y[idx-1])/(dx**2)/eps[1]
        
    # 2-3 boundary
    start = int((L[0]+L[1])/dx)
    for i in [start]:
        if internal_BC_type == '1':
            dy[i] = dy[i] + (2.0/dx/(eps[1]+eps[2]))*(D[2]*(-3*y[i]+4*y[i+1]-y[i+2]) - \
                                          D[1]*(3*y[i]-4*y[i-1]+y[i-2]))/2.0/dx
        else:
            dy[i] = 0
            
    # Inner third zone

    if internal_BC_type == '1':
        start = int((L[0]+L[1])/dx) + 1
    elif internal_BC_type == '2':
        idx = int((L[0]+L[1])/dx) + 1
        C_BC = (4*D[2]*y[idx]-D[2]*y[idx+1]-D[1]*y[idx-3]+4*D[1]*y[idx-2])/(3*D[1] + 3*D[2])
        dy[idx] = dy[idx] + D[2]*(y[idx+1] - 2*y[idx] + C_BC)/(dx**2)/eps[2]
        
        start = int((L[0]+L[1])/dx) + 2
    else:
        raise Exception('Unknown type of internal BC!')

    stop = int((L[0]+L[1]+L[2])/dx) - 1 - 1
    for i in range(start, stop+1):
        dy[i] = dy[i] + D[2]*(y[i+1] - 2*y[i] + y[i-1])/(dx**2)/eps[2]

    # Last inner point before exit (exit boundary C=0)
    start = int((L[0]+L[1]+L[2])/dx) - 1
    for i in [start]:
        dy[i] = dy[i] + D[2]*(- 2*y[i] + y[i-1])/(dx**2)/eps[2]

    # Pore-scale diffusion

    # Number of points in the middle zone
    start = int(L[0]/dx)
    stop = int((L[0]+L[1])/dx)
    N_mz = len(range(start, stop+1))

    # The last index of the reactor-scale grid points
    Idx_lr = int((L[0]+L[1]+L[2])/dx)

    # Number of points in the pore
    N_pore = int(Rp/dr) + 1

    for i in range(0,N_mz):
        start = Idx_lr + 1 + i*N_pore
        stop = start + N_pore - 2

        # Inner pore
        for j in range(start, stop+1):
            if micropore_geometry == 'linear':
                dy[j] = dy[j] + Dp*(y[j+1] - 2*y[j] + y[j-1])/(dr**2)

            elif micropore_geometry == 'spherical':
                dy[j] = dy[j] + Dp*((ri(j-start+1)+dr)*y[j+1] - 2*ri(j-start+1)*y[j] + \
                                    (ri(j-start+1)-dr)*y[j-1])/ri(j-start+1)/(dr**2)

            else:
                raise Exception('Unknown micropore geometry!')

        # Symmetry at the pore center
        for j in [stop+1]:
            if micropore_geometry == 'linear':
                dy[j] = dy[j] + Dp*(2*y[j-1] - 2*y[j])/(dr**2)

            elif micropore_geometry == 'spherical':
                dy[j] = dy[j] + 3*Dp*(2*y[j-1] - 2*y[j])/(dr**2)

            else:
                raise Exception('Unknown micropore geometry!')
                
    # Pore mouth boundary conditions

    # Index inside the middle reactor zone
    idx_r = lambda i: int(L[0]/dx) + i

    # Index of the corresponding pore mouth point
    idx_pm = lambda i: Idx_lr + 1 + i*N_pore

    if internal_BC_type == '1':
        idx_span = (0,N_mz)
    else:
        idx_span = (1,N_mz-1)

    for i in range(idx_span[0], idx_span[1]):
        if micropore_BC_type == 'kinetics':
            # The gas phase
            dy[idx_r(i)] = dy[idx_r(i)] - (1-eps[1])*(ka*y[idx_r(i)]*(Ns - y[idx_pm(i)]) + \
                                                      kd*y[idx_pm(i)])/eps[1]
                
            # The pore mouth
            dy[idx_pm(i)] = dy[idx_pm(i)] + (ka*y[idx_r(i)]*(Ns - y[idx_pm(i)]) - \
                                             kd*y[idx_pm(i)]) + \
                                             Dp*(y[idx_pm(i)+1] - y[idx_pm(i)])/(dr**2)

        elif micropore_BC_type == 'equilibrium':
            # The gas phase
            dy[idx_r(i)] = dy[idx_r(i)] + (1 - eps[1])*Dp*Sv*(y[idx_pm(i)+1] - KH*y[idx_r(i)])/dr/eps[1]

            # The pore mouth
            dy[idx_pm(i)] = KH*dy[idx_r(i)]

            # The point after the pore mouth
            if micropore_geometry == 'linear':
                dy[idx_pm(i)+1] = Dp*(y[idx_pm(i)+1+1] - 2*y[idx_pm(i)+1] + KH*y[idx_r(i)])/(dr**2)
            else:                
                dy[idx_pm(i)+1] = Dp*((ri(1)+dr)*y[idx_pm(i)+1+1] - 2*ri(1)*y[idx_pm(i)+1] + \
                                      (ri(1)-dr)*KH*y[idx_r(i)])/ri(1)/(dr**2)
                
        else:
            raise Exception('Unknown boundary conditions at the micropore mouth!')    
    
    return dy

def format_parameters(tree):

    # Read in the etree if a filename string is given
    if isinstance(tree, str):
        tree = etree.parse(tree)
    # Else, extract an etree if a tapper.experiment is given        
    elif isinstance(tree, Experiment):
        tree = tree.get_tree()
    else:
        pass

    opts = {}

    root = tree.getroot()
    # Only one (first) experiment can be handled for now 
    ex = root.findall('experiment')[0]
    
    if ex.find('geometry') is None or len(ex.find('geometry')) == 0:
        base = root
    else:
        base = ex

    zs = base.find('geometry').findall('zone')
    # assert that there are only three zones (implement one zone soon)
    assert(len(zs)==3)

    opts['length'], opts['porosity'], opts['D_ref'] = [], [], []

    opts['cross_section'] = float(zs[0].attrib['cross_section'])
    
    for i, z in enumerate(zs):
        opts['length'].append(float(z.attrib['length']))
        
        if len(z.findall('transport')) == 0:
            if base.find('geometry').find('transport') is not None:
                trans = base.find('geometry').find('transport')
            elif root.find('transport') is not None:
                trans = root.find('transport')
            else:
                raise Exception('No transport information for zone: %s' %
                                (tree.getpath(z),))
        else:
            try:
                trans = z.find('transport')
            except Exception as err:
                raise err('No transport information for zone: %s' %
                          (tree.getpath(z),))

        opts['porosity'].append(float(trans.attrib['porosity']))
        opts['D_ref'].append(float(trans.attrib['D_ref']))

    # assert that the middle zone has a material for which
    # pore diffusion kinetics is defined in the gloabl chemistry element
    # For now, pore diffusion kinetics in other zones and/or
    # other kinetics in any zone will be ignored.
    # For now, the first available catalyst from the second zone is used
    cat_name = zs[1].find('catalyst').attrib['name']
    mcat = zs[1].find('catalyst').attrib['mass']
    cat = root.find('chemistry').find('.//catalyst[@name="%s"]' % (cat_name))
    assert(len(cat.find('micropore'))>0)
    mpore = cat.find('micropore')
    
    opts['micropore_BC_type'] = mpore.find('BC').attrib['{http://www.w3.org/2001/XMLSchema-instance}type']
    if opts['micropore_BC_type']=='kinetics':
        opts['micropore_Ns'] = float(mpore.find('BC').attrib['Ns'])
        ka = float(mpore.find('BC').attrib['ka'])
        kd = float(mpore.find('BC').attrib['kd'])
    elif opts['micropore_BC_type']=='equilibrium':
        opts['micropore_Sv'] = float(mpore.find('BC').attrib['Sv'])
        KH = float(mpore.find('BC').attrib['KH'])

    opts['micropore_geometry'] = mpore.find('diffusion').attrib['geometry']
    opts['micropore_Rp'] = float(mpore.find('diffusion').attrib['Rp'])
    # For now, an isothermal value is expected.
    # In the future, to be extended to Arrhenius parameters
    Dp = float(mpore.find('diffusion').attrib['Dp'])
    
    if opts['micropore_BC_type']=='equilibrium':
        pars = zeros(2)
        pars[0] = Dp
        pars[1] = KH
    else:
        pars = zeros(3)
        pars[0] = Dp
        pars[1] = ka
        pars[2] = kd

    # handle one gas at a time, which means it must be marked in the tree
    gas = root.find('chemistry').find('.//gas[@micropore="true"]')
    mass = sum([int(x.attrib['n']) * float(el.attrib['mass'])
            for el in root.find('chemistry').findall('element')
            for x in gas.findall('element')
            if el.attrib['name'] == x.attrib['name']])
    opts['M'] = mass
    opts['gas'] = gas.attrib['name']
    
    ref_gas = root.find('chemistry').find('.//gas[@reference="true"]')
    ref_mass = sum([int(x.attrib['n']) * float(el.attrib['mass'])
                    for el in root.find('chemistry').findall('element')
                    for x in ref_gas.findall('element')
                    if el.attrib['name'] == x.attrib['name']])

    opts['Mref'] = ref_mass

    # inputSequence
    # For now, the first valve with declared Np is used and
    # pulsing from more than one valve cannot be accomodated.
    if ex.find('inputSequence') is None:
        base = root
    else:
        base = ex

    inSeq = base.find('inputSequence')
    valve = inSeq.find('.//valve[@Np]')
    Np_total = float(valve.attrib['Np'])
    fraction = float(valve.find('.//gas[@name="%s"]' % (gas.attrib['name'])).attrib['fraction'])
    opts['Np'] = Np_total*fraction
    opts['pTime'] = float(valve.attrib['pTime'])

    # outputSequence
    if ex.find('outputSequence') is None:
        base = root
    else:
        base = ex

    outSeq = base.find('outputSequence')
    opts['cTime'] = float(outSeq.attrib['cTime'])
    # For now, nSamples is equal to nPoints, which should be extended in the future
    opts['nSamples'] = float(outSeq.attrib['nPoints'])
    opts['cf'] = float(outSeq.find('.//response/gas[@name="%s"]' % (gas.attrib['name'])).attrib['cf'])

    # temperature
    opts['T'] = float(ex.find('.//field[@name="temperature"]').text)
    
    # general options
    opts['internal_BC_type'] = root.find('.//option[@key="internal_BC_type"]').attrib['value']
    opts['Tref'] = float(root.find('.//option[@key="T_ref"]').attrib['value'])
    opts['dx'] = 1.0/float(root.find('.//option[@key="integration"]').attrib['n_gridpoints_per_m'])
    opts['dr'] = 1.0/float(root.find('.//option[@key="micropore"]').attrib['n_gridpoints_per_m'])
    
    return pars, opts

def set_initial_conditions(opts):
    L = opts['length']
    dx = opts['dx']
    Rp = opts['micropore_Rp']
    dr = opts['dr']

    # Number of points in the middle zone
    start = int(L[0]/dx)
    stop = int((L[0]+L[1])/dx)
    N_mz = len(range(start, stop+1))

    # Number of the reactor-scale grid points
    N_r = int((L[0]+L[1]+L[2])/dx) + 1

    # Number of points in the pore
    N_pore = int(Rp/dr) + 1

    Ngp = N_r + N_mz*N_pore
    
    y0 = zeros(Ngp)
    
    return y0

def format_results(sol, pars, opts):
    T = opts['T']                                    # Temperature, K
    Tref = opts['Tref']                              # Reference temperature for Knudsen diffusivity, K
    
    L = opts['length']                               # List of zone lengths, m 
    dx = opts['dx']                                  # Reactor-scale grid step, m
    A = opts['cross_section']                        # Reactor cross-section, m^2
    eps = opts['porosity']                           # List of zone porosities
    
    D_ref = array(opts['D_ref'])                     # Reference Knudsen diffusivity, m^2/s 
    Mref = opts['Mref']                              # Reference molecular mass for Knudsen diffusivity, g/mol
    M = opts['M']                                    # Molecular mass of the diffusing gas, g/mol

    Np = opts['Np']                                  # Pulse size, mol
    tau_p = opts['pTime']                            # Pulse peak time (Gaussian), s

    Rp = opts['micropore_Rp']                        # Characteristic length of micropore diffusion, m
    dr = opts['dr']                                  # Micropore-scale grid step, m 
    micropore_BC_type = opts['micropore_BC_type']    # 'kinetics' or 'equilibrium'
    if micropore_BC_type=='equilibrium':
        Sv = opts['micropore_Sv']                        # Surface-to-volume ratio for the microporous particles, 1/m

    if micropore_BC_type=='kinetics':
        Ns = opts['micropore_Ns']                        # Total concentration of adsorption sites in the pore-mouth control volume, mol/m_s^3
    
    time = arange(0, opts['cTime'], opts['cTime']/opts['nSamples'])
    
    D = D_ref*(T*Mref/M/Tref)**0.5

    # Recovering the boundary concentration at the inlet
    sol[:,0] = ((2*Np*dx/D[0]/A)*(time*exp(-time/tau_p)/tau_p**2) + 4*sol[:,1] - sol[:,2])/3.0
    
    # Recovering the boundary concentrations between the reactor zones
    if opts['internal_BC_type'] == '2':
        idx = int(L[0]/dx)
        sol[:,idx] = (4*D[1]*sol[:,idx+1] - D[1]*sol[:,idx+2] - D[0]*sol[:,idx-2] + 4*D[0]*sol[:,idx-1])/(3*D[0] + 3*D[1])

        idx = int((L[0]+L[1])/dx)
        sol[:,idx] = (4*D[2]*sol[:,idx+1] - D[2]*sol[:,idx+2] - D[1]*sol[:,idx-2] + 4*D[1]*sol[:,idx-1])/(3*D[2] + 3*D[1])


    # Recovering the pore mouth surface concentration for the equilibrium BC

    # The last index of the reactor-scale grid points
    Idx_lr = int((L[0]+L[1]+L[2])/dx)

    # Number of points in the middle zone
    start = int(L[0]/dx)
    stop = int((L[0]+L[1])/dx)
    N_mz = len(range(start, stop+1))

    # Number of points in the pore
    N_pore = int(Rp/dr) + 1    

    # Index inside the middle reactor zone
    idx_r = lambda i: int(L[0]/dx) + i

    # Index of the corresponding pore mouth point
    idx_pm = lambda i: Idx_lr + 1 + i*N_pore
    
    if opts['micropore_BC_type'] == 'equilibrium':

        if opts['internal_BC_type'] == '1':
            idx_span = (0,N_mz)
        else:
            idx_span = (1,N_mz-1)

        for i in range(idx_span[0], idx_span[1]):
            KH = pars[1]
            sol[:,idx_pm(i)] = KH*sol[:,idx_r(i)]

    fexit = D[2]*A*sol[:,Idx_lr-1]/dx
    time = arange(0, opts['cTime'], opts['cTime']/opts['nSamples'])
    
    output = {}

    # Need to modify the Field.from_array method to simplify creating Fields from array
    data = np.zeros((1,len(time),1))
    data[0,:,0] = fexit*opts['cf']
    coords = (
        ('space', [1]),
        ('period', pd.MultiIndex.from_product(
            (np.array([1]),time),
             names=('pulse', 'time')
        )),
        ('amu', [str(opts['M'])])
    )
    output['signal'] = Field.from_array(data, coords=coords)

    gas_conc = np.zeros((Idx_lr+1,len(time),1))
    gas_conc[:,:,0] = sol[:,:Idx_lr+1].T
    coords = (
        ('space', np.arange(0,(Idx_lr+1)*dx,dx)),
        ('period', pd.MultiIndex.from_product(
            (np.array([1]),time),
             names=('pulse', 'time')
        )),
        ('gas_species', [opts['gas']])
    )
    output['gas_concentrations'] = Field.from_array(gas_conc, coords=coords)

    pore_conc = np.zeros((N_mz,len(time),N_pore))
    # Index of the corresponding pore mouth point
    idx_pm = lambda i: Idx_lr + 1 + i*N_pore
        
    for j in range(start,stop+1):
        pore_conc[j-start,:,:] = sol[:,idx_pm(j-start):idx_pm(j-start)+N_pore]

    coords = (
        ('space', np.arange(L[0],L[0]+L[1],dx)),
        ('period', pd.MultiIndex.from_product(
            (np.array([1]),time),
             names=('pulse', 'time')
        )),
        ('pore_species', [str(r) for r in np.arange(0,Rp+dr/2.0,dr)])
    )
    output['pore_concentrations'] = Field.from_array(pore_conc, coords=coords)

    # output['stats'] = Return stats as a separate entity
    
    return output

def simulate(pars, opts):

    #pars, opts = format_parameters(fname)
    mod = lambda t,y: model_equations(t, y, pars, opts)
    y0 = set_initial_conditions(opts)
    time = arange(0, opts['cTime'], opts['cTime']/opts['nSamples'])
    sol, stats = odeint(mod, y0, time, tfirst=True, full_output=True)

    output = format_results(sol, pars, opts)
    
    return output

def regress(pars0, opts, fexit):

    def _reformat(input_field):
        # Re-casting the only relevant pulse response as np.array
        amu = input_field['signal'].coords['amu'][0]
        output_array = input_field['signal'].data.sel(space=1,pulse=1,amu=amu).values
        return output_array
    
    def _sim_model(pars, t):
        #pars0, opts = format_parameters(fname)
        mod = lambda t,y: model_equations(t, y, pars, opts)
        y0 = set_initial_conditions(opts)
        time = arange(0, opts['cTime'], opts['cTime']/opts['nSamples'])
        sol, stats = odeint(mod, y0, time, tfirst=True, full_output=True)

        # The Field should be re-formatted differently for the ODR to inject
        output = _reformat(format_results(sol, pars, opts))

        return output

    #def perform_odr(x, y, xerr, yerr):
    def perform_odr(t, fexit):
        model = odr.Model(_sim_model)
        #mydata = odr.Data(x, y, wd=1./xerr, we=1./yerr)
        mydata = odr.Data(t, fexit)
        myodr = odr.ODR(mydata, model, beta0=pars0)
        output = myodr.run()
        return output

    t =  arange(0, opts['cTime'], opts['cTime']/opts['nSamples'])

    regression = perform_odr(t, fexit)
    fmod = _sim_model(regression.beta, t)

    return t, fexit, fmod, regression

if __name__=='__main__':
    # sys.argv[1]
    pass
