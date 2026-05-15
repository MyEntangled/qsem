import numpy as np
from scipy.special import jv, iv

def sin_cheb_appr(deg):
    res = np.zeros(deg+1)
    for i in range(1,deg+1,2):
        res[i] = 2*(-1)**(i//2)*jv(i,1)
    return res

def cos_cheb_appr(deg):
    res = np.zeros(deg+1)
    res[0] = jv(0,1)
    for i in range(2,deg+1,2):
        res[i] = 2*(-1)**(i//2)*jv(i,1)
    return res

def exp_cheb_appr(deg):
    res = np.array([2*iv(i,1) for i in range(deg+1)])
    res[0] = iv(0,1)
    return res
