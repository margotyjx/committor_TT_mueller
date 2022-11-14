"""
1, Initialize the tensor train format of Q 
"""


from scipy import integrate
import numpy as np
from numpy import exp
from scipy import special
from math import pi
# import xerus as xe
import pickle
import orth_pol_q

"""
xerus version
"""

# +
# # rank is the rank for the tensor train
# # d is the dimension
# # pol_deg is maximum degree of the orthogonal polynomials to connect
# # NOTE: we only consider the case when the number of orthogonal polynomials
# # are the same for each dimensions.
# def init_Q(rank,d,max_pol_deg):
#     rank_vec = [rank]*(d-1)
#     Q_init = xe.TTTensor.random([max_pol_deg+1]*d, rank_vec)
#     pickle.dump(Q_init, open("Q_init", 'wb')) 
    
#     return Q_init

# +
# def coef_ABP(coef):
#     length = len(coef)
#     rank = 1
#     rank_vec = [rank]*(length - 1)
#     A = xe.TTTensor.random([1]*length,rank_vec)
    
#     for i in range(length):
#         new_co = xe.Tensor.from_ndarray(np.array([[[coef[i]]]]))
#         A.set_component(i,new_co)
#     return A
# -

"""
numpy version
"""


def init_Q_np(rank,d,max_pol_deg):
    pol_deg = max_pol_deg+1
    Q = []
    for i in range(d):
        if i == 0:
            Q_temp = np.random.rand(pol_deg,rank).astype(np.float64)
        elif i == d-1:
            Q_temp = np.random.rand(pol_deg,rank).astype(np.float64)
        else:
            Q_temp = np.random.rand(rank,pol_deg,rank).astype(np.float64)
        Q.append(Q_temp)
        
        pickle.dump(Q, open("Q", 'wb')) 
    
    return Q

# +
# Q = init_Q_np(2,2,5)
# -


