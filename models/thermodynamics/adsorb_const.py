import numpy as np

def ka_hertz_knud(opts, temp, stick_coeff = 0.99):
    Ac = opts['AS concentration'] / opts['Surface per weight']
    return stick_coeff/Ac*opts['porosity']/(1-opts['porosity'])*(8.314*temp/2/np.pi/opts['M'])**(0.5)
