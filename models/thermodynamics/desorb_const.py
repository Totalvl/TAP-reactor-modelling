import numpy as np
E_bond = {"C2H6": -16.2, "C3H8": -29, "C4H10": -41, "C5H12": -51, "C6H14": -62, "C7H16": -70, "C8H18": -78, 
         "C2H4": -97.5, "C3H6": -107, "cis-C4H8": -101, "iC4H8": -92, "C5H10": -111, "C6H12": -117}
low_freq = {"C2H4": 126.5, "C3H6": 135.9, "cis-C4H8": 103.4, "iC4H8": 123.7, "alkane": 333.56}

def k_des_frenkel(gas, T):
    if gas in ['C2H6',"C3H8", "C4H10", "C5H12", "C6H14", "C7H16", "C8H18"]:
        char_freq = 299792458/(1/(low_freq["alkane"]*100))
    else:
        char_freq = 299792458/(1/(low_freq[gas]*100))
    return char_freq*np.exp(E_bond[gas]*1000/8.314/T)

def k_des_frenkel_manual(gas, T, E_b):
    if gas in ['C2H6',"C3H8", "C4H10", "C5H12", "C6H14", "C7H16", "C8H18"]:
        char_freq = 299792458/(1/(low_freq["alkane"]*100))
    else:
        char_freq = 299792458/(1/(low_freq[gas]*100))
    return char_freq*np.exp(E_b/8.314/T)

alpha_const = {"ZSM5": -16.5, "FAU": -8.6, "BEA": -10.8, "MOR": -13.8}
beta_const = {"ZSM5": -2.6, "FAU": -8.7, "BEA": -10.8, "MOR": -13.8}