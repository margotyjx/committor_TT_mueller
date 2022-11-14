from scipy import integrate
import numpy as np
from numpy import exp
from scipy import special
from math import pi
# import xerus as xe
import pickle
import orth_pol_q
import matplotlib.pyplot as plt


def eval_Q(beta,X,filename):
    # X should be row vectors where each columns represent the i-th dimension of the data tested 
    Q = pickle.load(open(filename, 'rb'))
    d = len(Q)
    pol_deg,rank = Q[0].shape
    max_pol_deg = pol_deg-1
    max_pol_deg_vec = [max_pol_deg]*d
    data_size = len(X)
    
    all_phi = orth_pol_q.all_pol_basis(max_pol_deg_vec,d,beta)
    Phi_x = np.zeros((d,pol_deg,data_size))
    
    
    for i in range(d):
        phi_i = all_phi[i]
        xi = X[:,i]
        for j in range(pol_deg):
            poly = phi_i[j]
            pol_x = poly(xi)
            Phi_x[i,j,:] = pol_x
            
    for pos in range(d):
        if pos == 0:
            Qi = Q[0]
            eval_ = np.einsum('ij,ik->jk',Qi,Phi_x[pos,:,:])
        elif pos == d-1:
            Qi = Q[d-1]
            eval_ = np.einsum('il,ji,jl->l',eval_,Qi,Phi_x[pos,:,:])
        else: 
            Qi = Q[pos]
            eval_ = np.einsum('il,ijk,jl->kl',eval_,Qi,Phi_x[pos,:,:])

    return eval_


