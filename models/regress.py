import numpy as np
import matplotlib.pyplot as plt
from . import equilibr as eq
# import models.microAD as mad
from . import adsdesNonLin as ad
from . import adsdesLin as adl
from . import equilibrCopy1 as eq1
from . import equilibrCopy2 as eq2
from . import equilibrCopy3 as eq3
from . import prototypes as pt
from . import Ansol as an
from . import SurfBar as sb
from . import Ansol_dless as andl
from models.prototypes import universal

from functools import partial
from scipy.optimize import least_squares
from scipy.stats.distributions import  t as Student_t


def simulation_num(t, pars, opts, model, Nx):
    dt = t[1]-t[0]
    T = t[-1]
    if model == 'Diff_3Z':
        opts['Dref zones'] = np.exp([pars[0], pars[1], pars[0]])
        return adl.pulse(pars=[0,0], opts=opts, T=T, dt=dt, Nx=Nx)['Flow'].to_numpy()
    elif model == 'pore_eq_lin':
        pars = np.exp(pars)
        return eq1.pulse(pars=pars, opts=opts, T=T, dt=dt, Nx=Nx)[1]
    elif model == 'rev_ads':
        pars = np.exp(pars)
        return ad.pulse(pars=pars, opts=opts, T=T, dt=dt, Nx=Nx)['Flow'].to_numpy()       
    elif model == 'surf_bar_full':
        print(pars)
        assert len(pars) == 5
        pars = np.exp(pars)
        return sb.pulse(pars=pars, opts=opts, T=T, dt=dt, Nx=Nx)[1]
    elif model == 'surf_bar_part':
        
        pars = np.exp(pars)
        assert len(pars) == 3
        assert opts['Surface adsorption/desorption constants'] is not None
        params = np.zeros(5)
        params[:2] = opts['Surface adsorption/desorption constants']
        params[2:] = pars[:]
        print(params)
        return sb.pulse(pars=params, opts=opts, T=T, dt=dt, Nx=Nx)[1]

def regress_num(y, t, opts, model, Nx=500, step = None, init_guess = None):
    print(step)
    regress = partial(simulation_num, opts = opts, model=model, Nx=Nx)
    
    def residuals(pars, y, t):
        print(max(y - regress(t, pars))/1e-9)
        return (y - regress(t, pars))/1e-9
    
    if model == 'Diff_3Z':
        if init_guess == None:
            init_guess = [-6.9, -9.2]
        bounds = ([-13.5, -13.5], [-4.5, -4.5])
    elif model == 'pore_eq_lin':
        if init_guess == None:
            init_guess = [0, -27.5]
        bounds = ([-12.0, -35.0], [7.0, -9.0])
    elif model == 'rev_ads':
        if init_guess == None:
            init_guess = [0.0, 0.0]
        bounds = ([-15.0, -15.0], [25.0, 25.0])      
    elif model == 'surf_bar_full':
        if init_guess == None:
            init_guess = [2.0, 4.0, -20.0, 2.0, 4.0]
        bounds = ([-15.0, -15.0, -35, -15.0, -15.0], [25.0, 25.0, -9.0, 25.0, 25.0])        
    elif model == 'surf_bar_part':
        if init_guess == None:
            init_guess = [-10.0, 2.3, 5.0]
        bounds = ([-35, -12.0, -12.0], [-9.0, 15.0, 15.0])          
        
    regr_par = least_squares(residuals,  x0 = init_guess, jac = '2-point', bounds = bounds, 
                             method = 'dogbox', loss = 'soft_l1', max_nfev = 10000, 
                             diff_step = step, tr_solver = 'exact', 
                             args = (y, t))
    return regr_par['x']
#     return residuals(y, t, pars)

def simulation_TM(t, pars, opts, model):
    dt = t[1]-t[0]
    T = t[-1]
    return pt.pulse(pars=pars, opts=opts, model = model, T=T, dt=dt)

def regress_TM(y, t, opts, model, step = None, init_guess = None):
    
    regress = partial(simulation_TM, opts = opts, model=model)
    
    def residuals(pars, y, t):
        print(max(y - regress(t, np.power(10.0, pars))))
        return (y - regress(t, np.power(10.0, pars)))
    if model == 'Diff_3Z':
        if init_guess == None:
            init_guess = [-3, -4]
        bounds = ([-7, -8], [-1, -1])
    elif model == 'pore_eq_lin':
        if init_guess == None:
            init_guess = [0, -12]
        bounds = ([-5.0, -17.0], [5.0, -8.0])
    elif model == 'rev_ads':
        if init_guess == None:
            init_guess = [0.0, 0.0]
        bounds = ([-5.0, -5.0], [10.0, 10.0])      
    elif model == 'surf_bar_full':
        if init_guess == None:
            init_guess = [0.0, 0.0, 0.0, 0.0, -12.0]
        bounds = ([-5.0, -5.0, -5.0, -5.0, -17.0], [10.0, 10.0, 10.0, 10.0, -8.0])        
    elif model == 'surf_bar_part':
        if init_guess == None:
            init_guess = [0.0, 0.0, -12.0]
        bounds = ([-5.0, -5.0, -17.0], [10.0, 10.0, -8.0])
    elif model == 'SBIALP_full':
        if init_guess == None:
            init_guess = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -12.0]
        bounds = ([-15.0, -15.0, -15.0, -15.0, -15.0, -15.0, -17.0], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -8.0])
    elif model == 'SBIALP_part':
        if init_guess == None:
            init_guess = [0.0, 0.0, 0.0, 0.0, -12.0]
        bounds = ([-15.0, -15.0, -15.0, -15.0, -15.0, -15.0, -17.0], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -8.0])
    # add max_nfev = 1000, to function below to limit number of calls     
    regr_par = least_squares(residuals,  x0 = init_guess, jac = '2-point', bounds = bounds, 
                             method = 'dogbox', loss = 'soft_l1',  
                             diff_step = step, tr_solver = 'exact', 
                             args = (y, t))
    return regr_par['x']

def simulation_an(t, pars, opts, model, Nx):
    dt = t[1]-t[0]
    T = t[-1]
    if model == 'analytical_5P':
        pars = np.concatenate((pars[:4], [0,0,pars[-1]]))
        model = 'analytical_7P'
    
    pars=np.power(10, pars)
    return andl.pulse(pars=pars, opts=opts, model = model, T=T, dt=dt, Nx=Nx)[1]

def regress_an(y, t, opts, model, Nx=500, step = None, init_guess = None):
    
    regress = partial(simulation_an, opts = opts, model=model, Nx=Nx)
    
    def residuals(pars, y, t):
        return (y - regress(t, pars))

    if model == 'rev_ads':
        if init_guess is None:
            init_guess = [0.0, 0.0]
        bounds = ([-10.0, -10.0], [10.0, 10.0])
    elif model == 'analytical_EqMP':
        if init_guess is None:
            init_guess = [0.0, 5.0]
        bounds = ([-10.0, -10.0], [10.0, 10.0])
    elif model == 'analytical_DAMP':
        if init_guess is None:
            init_guess = [0.0, 0.0, 0.0]
        bounds = ([-10, -10.0, -10], [10.0, 10.0, 10.0])      
    elif model == 'analytical_7P':
        if init_guess is None:
            init_guess = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        bounds = ([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    elif model == 'analytical_5P':
        if init_guess is None:
            init_guess = [0.0, 0.0, 0.0, 0.0, 0.0]
        bounds = ([-10.0, -10.0, -10.0, -10.0, -10.0], [10.0, 10.0, 10.0, 10.0, 10.0])
    elif model == 'analytical_DAMPint':
        if init_guess is None:
            init_guess = [0.0, 0.0, 0.0, 0.0, 0.0]
        bounds = ([-10.0, -10.0, -10.0, -10.0, -10.0], [10.0, 10.0, 10.0, 10.0, 10.0])
        
  
    # add max_nfev = 1000, to function below to limit number of calls     
    regr_par = least_squares(residuals,  x0 = init_guess, jac = '2-point', bounds = bounds, 
                             method = 'dogbox', loss = 'soft_l1',  
                             diff_step = step, tr_solver = 'exact', 
                             args = (y, t))
    return regr_par['x']

def simulate_multiresponse_TM(times,TD_pars,  opts, temps,  model, inf_time): # for regression of the options instead of parameter change the sequence of TD_pars and opts
    responses = []
    for n, temp in enumerate(temps):
        if model == 'Diff_1Z':
            opts['Dref zones'] = np.power(10.0, [TD_pars[0], TD_pars[0], TD_pars[0]]) # IMPORTANT!!! If model is selected as Diff_1Z the pars overwrite opts['Dref']
            pars = None
        elif model == "Diff_3Z":
            opts['Dref zones'] = np.power(10.0, [TD_pars[0], TD_pars[1], TD_pars[0]]) # IMPORTANT!!! If model is selected as Diff_3Z the pars overwrite opts['Dref']
            pars = None
        elif model == "Diff_CatZ":
            opts['Dref zones'][1] = 10.0**TD_pars[0] # IMPORTANT!!! If model is selected as Diff_3Z the pars overwrite opts['Dref']
            pars = None
        elif model == 'surf_bar_part':
            TD_pars1 = np.insert(TD_pars, [1,  6], [0, TD_pars[4]])
            pars = [np.power(10.0, TD_pars1[m])*np.exp(-TD_pars1[m+1]*1e3/8.314/(temp+273)) for m in range(len(TD_pars1)) if m%2 ==0]
        elif model == 'rev_ads' or model == 'rev_ads_MPD' or model == 'rev_ads_MPD_int_ad' or model == 'RA_MPD_IA_part' or model == 'surf_bar_full' or model == 'surf_bar_part' or model == 'SBIALP_full':
            TD_pars1 = np.insert(TD_pars, 1, 0)
            if model == 'rev_ads_MPD_int_ad':
                TD_pars1 = np.insert(TD_pars1, 5, 0)
            pars = [np.power(10.0, TD_pars1[m])*np.exp(-TD_pars1[m+1]*1e3/8.314/(temp+273)) for m in range(len(TD_pars1)) if m%2 ==0]
        elif model =='pore_eq_lin':
            pars = [np.power(10.0, TD_pars[m])*np.exp(-TD_pars[m+1]*1e3/8.314/(temp+273)) for m in range(len(TD_pars)) if m%2 ==0]
        else: 
            raise NameError('Model name does not exist')
        opts['T'] = 273+temp
        if inf_time !=0:
            tid = np.linspace(0, inf_time, int(round(inf_time/(times[n][1] - times[n][0]))))
            response = simulation_TM(tid, pars, opts, model)[1]
            response = response[:len(times[n])]
        else:
            response = simulation_TM(times[n], pars, opts, model)[1]
        
        responses.append(response)
    return np.array(responses)
        
def regress_multiresponse_TM(flows, times, temps, opts, model, step = None, init_guess = None, area_normalized = False,
                             height_normalized = False, tail_enhance = False, inf_time = 0, exp_data_range = None):
    concat_flows = np.array([])
    concat_times = np.array([])
    for n, flow in enumerate(flows):
        dt = times[n][1] - times[n][0]
        if area_normalized is True:
            flow = flow/np.trapz(flow, dx=dt, axis=0)
        elif height_normalized is True:
            flow = flow/np.mean(flow[np.argmax(flow)-3:np.argmax(flow)+3])
        if type(exp_data_range) is int and exp_data_range > 0:
            flow = flow[:exp_data_range]
        if tail_enhance == 'forward':
            concat_flows = np.append(concat_flows, flow*times[n]/dt)
        elif tail_enhance == 'backward':
            concat_flows = np.append(concat_flows, flow/(times[n]+dt))
        else:
            concat_flows = np.append(concat_flows, flow)
        concat_times = np.append(concat_times, times[n])
        
    regress = partial(simulate_multiresponse_TM, temps = temps, opts = opts, model=model, inf_time = inf_time)
    
    def residuals(TD_pars, y, t):
        print(TD_pars)
        sim_pulses = regress(times, TD_pars)
        concat_sim_pulses = np.array([])
        for m, pulse in enumerate(sim_pulses):
            dt = times[m][1] - times[m][0]
            if area_normalized is True:
                pulse = pulse/np.trapz(pulse, dx=dt, axis=0)
            elif height_normalized is True:
                pulse = pulse/np.mean(pulse[np.argmax(pulse)-3:np.argmax(pulse)+3])
            if type(exp_data_range) is int and exp_data_range > 0:
                pulse = pulse[:exp_data_range]                
            if tail_enhance == 'forward':
                concat_sim_pulses = np.append(concat_sim_pulses, pulse*times[m]/dt)
            elif tail_enhance == 'backward':
                concat_sim_pulses = np.append(concat_sim_pulses, pulse/(times[m]+dt))
            else:
                concat_sim_pulses = np.append(concat_sim_pulses, pulse)
            
        return (concat_flows - concat_sim_pulses)
    if model == 'Diff_1Z':
        if init_guess is None:
            init_guess = [-3.0]
        bounds = ([-9.0], [2.0])
    elif model == 'Diff_CatZ':
        if init_guess is None:
            init_guess = [-4.0]
            bounds = ([-9.0], [-2.0])
    elif model == 'Diff_3Z':
        if init_guess is None:
            init_guess = [-2.5, -3.6]
        bounds = ([-5.0, -5.0], [-1.0, -1.0])
    elif model == 'rev_ads':
        if init_guess is None:
            init_guess = [6.5, 12.5, 31.0]
        bounds = ([-10.0, -10.0, 0.0], [12.7, 13.0, 53.0])
        #             init_guess = [8.00, 13.0, 41.0]
        # bounds = ([0.0, 12.7, 0.0], [12.0, 13.0, 110.0])
    elif model == 'pore_eq_lin':
        if init_guess is None:
            init_guess = [-4.8, -54.0, -2.5, 60.0]
        bounds = ([-13.0, -73.0, -12.0, 15.0], [0.0, -29.0, -5.0, 100.0])
    elif model == 'rev_ads_MPD':
        if init_guess is None:
            init_guess = [6.3, 13., 51, -8.5, 20.0]
        bounds = ([0.0, 0.0, 49.0, -20.0, 0.0], [12.0, 13.0, 100.0, -5.0, 30.0])  
    elif model == 'rev_ads_MPD_int_ad':
        if init_guess is None:
            init_guess = [8.5, 12.7, 60.0, 8.5, 12.7, 60.0, -8.5, 20.0]
        bounds = ([7.8, 12.7, 0.0, 4.0, 12.7, 0.0, -15.0, 0.0], [12.0, 13.0, 100.0, 12.7, 13.0, 100.0, -4.0, 50.0])
    elif model == 'RA_MPD_IA_part':
        if init_guess is None:
            init_guess = [8.5, 12.7, 51.0, -8.5, 25.0]
        bounds = ([6.5, 12.7, 0.0, -14.0, 0.0], [12.0, 13.0, 105.0, -5.0, 100.0])
    elif model == 'surf_bar_full':
        if init_guess is None:
            init_guess = [7.5, 13.0, 51.0, 6.0, 25.0, 8.00, 25.0, -8.2, 25.0]
        bounds = ([7.2, 12.7,50.0, 0.0, 0.0, 0.0, 0.0, -14.0, 0.0], [7.5, 13.0, 120.0, 15.0, 150.0, 15.0, 150.0, -3.0, 150.0])        
    elif model == 'surf_bar_part':
        if init_guess is None:
            init_guess = [7.5, 13.0, 51.0, 6.0, 25.0, 6.0, -8.5, 25.0]
        bounds = ([7.0, 12.7, 50.0, 0.0, 0.0, 0.0, -14.0, 0.0], [8.0, 13.0, 55.0, 15.0, 150.0, 15.0, -3.0, 150.0])
    elif model == 'SBIALP_full':
        if init_guess is None:
            init_guess = [4.0, 12.0, 90.0, 5.0, 50.0, 12.0, 50.0, 4.0, 20.0, 12.0, 90.0, -11.0, 25.0]
        bounds = ([-5.0, -5.0, 0.0, -5.0, 0.0, -5.0, 0.0, -5.0, 0.0, -5.0, 0.0, -17.0, 0.0], 
                  [10.0, 15.0, 250.0, 10.0, 250.0, 15.0, 250.0, 10.0, 250.0, 15.0, 250.0, -8.0, 40.0])     
    elif model == 'SBIALP_part':
        if init_guess is None:
            init_guess = [5.0, 50.0, 12.0, 50.0, 4.0, 20.0, 12.0, 90.0, -11.0, 25.0]
        bounds = ([-5.0, 0.0, -5.0, 0.0, -5.0, 0.0, -5.0, 0.0, -17.0, 0.0], 
                  [10.0, 250.0, 15.0, 250.0, 10.0, 250.0, 15.0, 250.0, -8.0, 40.0]) 
    else:
        print("Unknown model")
        
  
    # add max_nfev = 1000, to function below to limit number of calls     
    regr_par = least_squares(residuals,  x0 = init_guess, jac = '2-point', bounds = bounds, 
                             method = 'dogbox', loss = 'soft_l1',  
                             diff_step = step, tr_solver = 'exact', 
                             args = (concat_flows, concat_times))
    
    def errFit(hess_inv, resVariance):
        return np.sqrt(np.diag( hess_inv * resVariance))
    se = errFit(np.linalg.inv(np.dot(regr_par.jac.T, regr_par.jac)), (residuals(regr_par.x, concat_flows, concat_times)**2).sum()/(len(concat_flows)-len(init_guess) ) ) 
    sT = Student_t.ppf(1.0 - 0.05/2.0,len(concat_flows)-len(regr_par.x))
    regr_par['CI'] = se * sT
    
    return regr_par
    
def regress_multiresponse_TM_ULT(flows, times, temps, opts, TD_pars, model, opts_name, step = None, init_guess = None, bounds = None, 
                                 area_normalized = False, height_normalized = False, inf_time = 0):
    concat_flows = np.array([])
    concat_times = np.array([])
    for n, flow in enumerate(flows):
        dt = times[n][1] - times[n][0]
        if area_normalized is True:
            flow = flow/np.trapz(flow, dx=dt, axis=0)
        elif height_normalized is True:
            flow = flow/np.mean(flow[np.argmax(flow)-3:np.argmax(flow)+3])
        concat_flows = np.append(concat_flows, flow)
        concat_times = np.append(concat_times, times[n])
        
        
    regress = partial(simulate_multiresponse_TM, TD_pars=TD_pars, temps = temps, model=model, inf_time = inf_time)
    
    def residuals(opt, y, t):
        print(opt)
        opts[opts_name] = 10**(-opt)
        sim_pulses = regress(times, opts)
        concat_sim_pulses = np.array([])
        for m, pulse in enumerate(sim_pulses):
            dt = times[m][1] - times[m][0]
            if area_normalized is True:
                pulse = pulse/np.trapz(pulse, dx=dt, axis=0)
            elif height_normalized is True:
                pulse = pulse/np.mean(pulse[np.argmax(pulse)-3:np.argmax(pulse)+3])
            concat_sim_pulses = np.append(concat_sim_pulses, pulse) 
        return (concat_flows - concat_sim_pulses)
    if init_guess is None:
        init_guess = 1
    if bounds is None:
        bounds = [8,11]
    # add max_nfev = 1000, to function below to limit number of calls     
    regr_par = least_squares(residuals,  x0 = init_guess, jac = '2-point', bounds = bounds, 
                             method = 'dogbox', loss = 'soft_l1',  
                             diff_step = step, tr_solver = 'exact', 
                             args = (concat_flows, concat_times))
    return regr_par
   
def Roelant_noize(time,pulse):
  sigma = float(np.random.normal(0.00497,0.00029/5, 1))
  zeta = float(np.random.normal(0.0434,0.0002/5, 1))
  alpha1 = float(np.random.normal(0.0281,0.0003/5, 1))
  alpha3 = float(np.random.normal(0.00266,0.00039/5, 1))
  A = float(np.random.normal(0.0,1, 1))
  theta1 = np.random.random()*2*np.pi
  theta2 = np.random.random()*2*np.pi
  theta3 = np.random.random()*2*np.pi
  beta2 = 0.00166
  beta3 = 0.0027
  noize = sigma * np.random.normal(0,1, len(pulse)) + zeta * A * pulse + alpha1*pulse*np.cos(2*np.pi*60*time+theta1)\
          + beta2*np.cos(2*np.pi * 70 * time + theta2) + (alpha3*pulse+beta3)*np.cos(2*np.pi * 120 * time+theta3)
  return noize   
 
def bootstrap(flows, times, temps, opts, pars, model, step = None, area_normalized = False,
                             height_normalized = False, inf_time = 0):
    plt.figure(figsize=(12, 8))
    sim_pulses = simulate_multiresponse_TM(times, pars,  opts, temps,  model, inf_time)
    
    ps = []
    for i in range(1000):
       flows_emul = flows.copy()
       sim_pulses_emul = sim_pulses.copy()
       for n, flow in enumerate(flows_emul):
            dt = times[n][1] - times[n][0]
            if area_normalized is True:
                 flow = flow/np.trapz(flow, dx=dt, axis=0)
                 sim_pulses_emul[n] = sim_pulses_emul[n]/np.trapz(sim_pulses_emul[n], dx=dt, axis=0)
            if height_normalized is True:
                 flow = flow/np.mean(flow[np.argmax(flow)-3:np.argmax(flow)+3])
                 sim_pulses_emul[n] = sim_pulses_emul[n]/np.mean(sim_pulses_emul[n][np.argmax(sim_pulses_emul[n])-3:np.argmax(sim_pulses_emul[n])+3])
            std = np.std(flow - sim_pulses_emul[n])
            flows_emul[n] = flow + Roelant_noize(times[n], flow)
            plt.plot(times[n], flows_emul[n])
            plt.xlim(0,3)
       plt.show()
       ps.append(regress_multiresponse_TM(flows = flows, times = times, temps = temps, opts = opts, model = model,
                                   step = step, init_guess = pars,
                                   area_normalized = area_normalized, height_normalized = height_normalized, inf_time = inf_time))
    ps = array(ps)
    mean_params = np.mean(ps,axis=0)
    std_params = np.std(ps,axis=0)

    return mean_params, std_params  
   
   

