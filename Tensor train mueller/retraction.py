from scipy import integrate
import numpy as np
from numpy import exp
from scipy import special
from math import pi
from scipy import interpolate


def fun_indices(k4,k5,k6,rank,pol_deg):
    # pol_deg is the number of orthogonal basis polynomials
    indice = k4*pol_deg*rank + k5*rank + k6
    return indice 


def identity_tensor(r_pre,pol_deg,r_curr):
    m = r_pre*pol_deg*r_curr
    I = np.zeros((r_pre,pol_deg,r_curr,m))
    for k4 in range(r_pre):
        for k5 in range(pol_deg):
            for k6 in range(r_curr):
                indice = fun_indices(k4,k5,k6,r_curr,pol_deg)
                I[k4,k5,k6,indice] = 1
    return I



# +
# A = identity_tensor(2,5,2)
# v = np.random.rand(2*5*2,)
# print(v)
# print(A@v)
# print((A@v).reshape(2*5*2))
# -
def identity_tensor_first(pol_deg,r_curr):
    m = pol_deg*r_curr
    I = np.zeros((pol_deg,r_curr,m))
    for k5 in range(pol_deg):
        for k6 in range(r_curr):
            indice = fun_indices(0,k5,k6,r_curr,pol_deg)
            I[k5,k6,indice] = 1
    return I 





