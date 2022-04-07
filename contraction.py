"""
contains the function to do the contraction to obtain H^k, H^A, H^B
h^B
"""


from scipy import integrate
import numpy as np
from numpy import exp
from scipy import special
from math import pi
# import xerus as xe
import pickle
import orth_pol_q,orth_pol_px
from scipy.special import roots_legendre


# +
class Component:
    def __init__(self,beta,d,max_pol_deg,max_pol_deg_px):
        self.beta = beta
        self.max_pol_deg = max_pol_deg
        self.d = d
        self.max_pol_deg_vec = [max_pol_deg]*d
        self.max_pol_deg_vec_px = [max_pol_deg_px]*2 + [max_pol_deg]*(d-2)
        self.pol_deg = self.max_pol_deg + 1
        self.pol_deg_px = max_pol_deg_px+1
        
        
        self.a1 = -1.5
        self.a2 = 1
        self.b1 = -0.5
        self.b2 = 2
        
        self.min = -1
        self.max = 1
        
        self.RA = 0.1
        self.RB = 0.1
        self.N_quad = 160
        self.set_A = np.array([-0.558,1.441])
        self.set_B = np.array([0.623,0.028])
        
        self.all_phi = orth_pol_q.all_pol_basis(self.max_pol_deg_vec,d,beta)

        self.pab = orth_pol_px.orth_pab(beta,d,max_pol_deg,max_pol_deg_px)    
        
    
    # this turn the poly1d into a symbolic function for integration
    def function_pi(self,poly_k):
    # k goes from 0 to j-1 in the for loop 
        def fun_pi(i1,pol_i1):
            return lambda x: (x**i1)*pol_i1

        def summer_funcs(args):
            return lambda x: sum(fun_pi(*a)(x) for a in args)

        arguments = []
        for i1 in range(len(poly_k)+1): # len(poly_k) = max_pol_deg
            deg = i1
            pol_i1 = poly_k[i1]
            arguments.append((deg,pol_i1))

        p_i = summer_funcs(arguments)

        return p_i
    
    # i,j are vector of indices of i1,...,id and j1,...,jd 
    # write it as a tensor train operator format where dimension is i_k x i_j
    # rank 1 for all core
    def Hk(self,indice_k):
        Hk = []
        for k in range(1,self.d):
            if k == 1:
                temp_Hk_0 = np.zeros((1,self.pol_deg,self.pol_deg,self.pol_deg_px))
                temp_Hk_1 = np.zeros((self.pol_deg_px,self.pol_deg,self.pol_deg,1))

                psi_curr,P_curr = self.pab.orth_p(k)

                psi_curr_0 = psi_curr[0]
                psi_curr_1 = psi_curr[1]

                P_curr_0 = P_curr[0] # both are matrices of size pol_deg_px by pol_deg_px
                P_curr_1 = P_curr[1] 

                for j in range(self.pol_deg):
                    phi_curr_j = self.all_phi[k][j]
                    phi_curr_jd = np.polyder(phi_curr_j)
                    phi_curr_j = self.function_pi(phi_curr_j)
                    phi_curr_jd = self.function_pi(phi_curr_jd)

                    for i in range(self.pol_deg):
                        phi_curr_i = self.all_phi[k][i]
                        phi_curr_id = np.polyder(phi_curr_i)
                        phi_curr_i = self.function_pi(phi_curr_i)
                        phi_curr_id = self.function_pi(phi_curr_id)
                        
                        if indice_k == 0:
                            for n in range(len(P_curr_0)):
                                Pi_0 = P_curr_0[n,:]
                                psi_0 = psi_curr_0[n]

                                func = lambda x: phi_curr_id(x)*phi_curr_jd(x)*psi_0(x)

#                                 int_result,err = integrate.quad(func,self.min,self.max)
                                int_result = self.gauss_quad(func,self.min,self.max,self.N_quad)
                                temp_Hk_0[0,i,j,:] += Pi_0*int_result
        

                            for m in range(len(P_curr_1)):
                                Pi_1 = P_curr_1[:,m]
                                psi_1 = psi_curr_1[m]

                                func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi_1(x)

#                                 int_result,err = integrate.quad(func,self.min,self.max)
                                int_result = self.gauss_quad(func,self.min,self.max,self.N_quad)

                                temp_Hk_1[:,i,j,0] += Pi_1*int_result
        
                
                        elif indice_k == 1:
                            for n in range(len(P_curr_0)):
                                Pi_0 = P_curr_0[n,:]
                                psi_0 = psi_curr_0[n]

                                func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi_0(x)

#                                 int_result,err = integrate.quad(func,self.min,self.max)
                                int_result = self.gauss_quad(func,self.min,self.max,self.N_quad)
                                temp_Hk_0[0,i,j,:] += Pi_0*int_result

                            for m in range(len(P_curr_1)):
                                Pi_1 = P_curr_1[:,m]
                                psi_1 = psi_curr_1[m]

                                func = lambda x: phi_curr_id(x)*phi_curr_jd(x)*psi_1(x)

#                                 int_result,err = integrate.quad(func,self.min,self.max)
                                int_result = self.gauss_quad(func,self.min,self.max,self.N_quad)
                                temp_Hk_1[:,i,j,0] += Pi_1*int_result
                    
                        else: 
                            for n in range(len(P_curr_0)):
                                Pi_0 = P_curr_0[n,:]
                                psi_0 = psi_curr_0[n]

                                func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi_0(x)

#                                 int_result,err = integrate.quad(func,self.min,self.max)
                                int_result = self.gauss_quad(func,self.min,self.max,self.N_quad)
                                temp_Hk_0[0,i,j,:] += Pi_0*int_result

                            for m in range(len(P_curr_1)):
                                Pi_1 = P_curr_1[:,m]
                                psi_1 = psi_curr_1[m]

                                func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi_1(x)

#                                 int_result,err = integrate.quad(func,self.min,self.max)
                                int_result = self.gauss_quad(func,self.min,self.max,self.N_quad)
                                temp_Hk_1[:,i,j,0] += Pi_1*int_result
                    
                Hk.append(temp_Hk_0)
                Hk.append(temp_Hk_1)
            
            else:
                if k != indice_k:
                    temp_Hk = np.zeros((1,self.pol_deg,self.pol_deg,1))
                    for j in range(self.pol_deg):
                        phi_curr_j = self.all_phi[k][j]
                        phi_curr_j = self.function_pi(phi_curr_j)
                        for i in range(self.pol_deg):
                            phi_curr_i = self.all_phi[k][i]
                            phi_curr_i = self.function_pi(phi_curr_i)

                            for m in range(len(P_curr)):
                                Pi = P_curr[m]
                                psi = psi_curr[m]
                                psi = self.function_pi(psi)

                                func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi(x)

#                                 int_result,err = integrate.quad(func,self.min,self.max)
                                int_result = self.gauss_quad(func,self.min,self.max,self.N_quad)
                                temp_Hk[0,i,j,0] += Pi*int_result
                    
                else: # k == indice_k
                    temp_Hk = np.zeros((1,self.pol_deg,self.pol_deg,1))
                    for j in range(self.pol_deg):
                        phi_curr_j = self.all_phi[k][j]
                        phi_curr_j = np.polyder(phi_curr_j)
                        phi_curr_j = self.function_pi(phi_curr_j)
                        for i in range(self.pol_deg):
                            phi_curr_i = self.all_phi[k][i]
                            phi_curr_i = np.polyder(phi_curr_i)
                            phi_curr_i = self.function_pi(phi_curr_i)

                            for m in range(len(P_curr)):
                                Pi = P_curr[m]
                                psi = psi_curr[m]
                                psi = self.function_pi(psi)

                                func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi(x)

#                                 int_result,err = integrate.quad(func,self.min,self.max)
                                int_result = self.gauss_quad(func,self.min,self.max,self.N_quad)
                                temp_Hk[0,i,j,0] += Pi*int_result
                    
                Hk.append(temp_Hk)  
        return Hk    
                    
                    

    
#    
    
    #easier version
    def HA(self):
        HA = []
        new_set_A = self.trans_into_standard_int(self.a1,self.a2,self.b1,self.b2,self.set_A)
        psi_a_currk,ak = self.pab.orth_ab(self.RA,new_set_A)
        
        for k in range(self.d):
            temp_HA = np.zeros((1,self.pol_deg,self.pol_deg,1))
            psi_a_curr = psi_a_currk[k]
           
            for j in range(self.pol_deg):
                phi_curr_j = self.all_phi[k][j]
                phi_curr_j = self.function_pi(phi_curr_j)
                for i in range(self.pol_deg):
                    phi_curr_i = self.all_phi[k][i]
                    phi_curr_i = self.function_pi(phi_curr_i)

                    func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi_a_curr(x)

                    int_result = self.gauss_quad(func,self.min,self.max,self.N_quad)
#                     int_result,err = integrate.quad(func,self.min,self.max)
        
                    temp_HA[0,i,j,0] = ak[k]*int_result
            HA.append(temp_HA)
        return HA

    
    # easier version
    def HB(self):
        HB = []
        new_set_B = self.trans_into_standard_int(self.a1,self.a2,self.b1,self.b2,self.set_B)
        psi_b_currk,bk = self.pab.orth_ab(self.RB,new_set_B)
        
        for k in range(self.d):
            temp_HB = np.zeros((1,self.pol_deg,self.pol_deg,1))
            psi_b_curr = psi_b_currk[k]
            
            for j in range(self.pol_deg):
                phi_curr_j = self.all_phi[k][j]
                phi_curr_j = self.function_pi(phi_curr_j)
                for i in range(self.pol_deg):
                    phi_curr_i = self.all_phi[k][i]
                    phi_curr_i = self.function_pi(phi_curr_i)

                    func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi_b_curr(x)
                    
#                     int_result,err = integrate.quad(func,self.min,self.max)
                    int_result = self.gauss_quad(func,self.min,self.max,self.N_quad)

                    temp_HB[0,i,j,0] = bk[k]*int_result
            HB.append(temp_HB)
        return HB
    
    # easier version
    def hb(self):
        hb = []
        new_set_B = self.trans_into_standard_int(self.a1,self.a2,self.b1,self.b2,self.set_B)
        psi_b_currk,bk = self.pab.orth_ab(self.RB,new_set_B)
        
        for k in range(self.d):
            temp_hb = np.zeros((1,self.pol_deg,1))
            psi_b_curr = psi_b_currk[k]
            
            for i in range(self.pol_deg):
                phi_curr_i = self.all_phi[k][i]
                phi_curr_i = self.function_pi(phi_curr_i)

                func = lambda x: phi_curr_i(x)*psi_b_curr(x)
                
#                 int_result,err = integrate.quad(func,self.min,self.max)

                int_result = self.gauss_quad(func,self.min,self.max,self.N_quad)
                temp_hb[0,i,0] = bk[k]*int_result
            hb.append(temp_hb)
        return hb
    
    def gauss_quad(self,func,left,right,N_quad):
        [x_quad, w_quad] = roots_legendre(N_quad)
        x_quad_new = left + (right-left)/2*(x_quad+1)
        coef = (right - left)/2
        summation = 0
        approx = sum(w_quad*func(x_quad))
        
        return approx
    
    def trans_into_standard_int(self,a1,a2,b1,b2,pts):
        new_pts = np.copy(pts)
        new_pts[0] = (2*new_pts[0] - (a1 + a2))/(a2 - a1)
        new_pts[1] = (2*new_pts[1] - (b1 + b2))/(b2 - b1)
        
        return new_pts
    
    


# +
# test = Component(beta = 1/30,d = 2,max_pol_deg=2)
# Hk_test = test.Hk(0)
# # print(Hk_test[1].shape)
# -

"""
The case when basis function used are exp(...)
"""


# +
class Component_easier:
    def __init__(self,beta,d,max_pol_deg,min_interval = -1,max_interval = 1):
        self.beta = beta
        self.max_pol_deg = max_pol_deg
        self.d = d
        self.max_pol_deg_vec = [max_pol_deg]*d
        self.pol_deg = self.max_pol_deg + 1
        self.min_interval = min_interval
        self.max_interval = max_interval
        
        
        self.all_phi = orth_pol_q.all_pol(self.max_pol_deg_vec,d,beta,a = -1,b = 1)

        self.pab = orth_pol_px.orth_pab_easier(beta,d,max_pol_deg)
        self.psi,self.P = self.pab.orth_p()
        self.psi_a,self.ak = self.pab.orth_ab(sigma=10,c=-1)
        self.psi_b,self.bk = self.pab.orth_ab(sigma=10,c=1)
        
        
        
    
    # this turn the poly1d into a symbolic function for integration
    def function_pi(self,poly_k):
    # k goes from 0 to j-1 in the for loop 
        def fun_pi(i1,pol_i1):
            return lambda x: (x**i1)*pol_i1

        def summer_funcs(args):
            return lambda x: sum(fun_pi(*a)(x) for a in args)

        arguments = []
        for i1 in range(len(poly_k)+1): # len(poly_k) = max_pol_deg
            deg = i1
            pol_i1 = poly_k[i1]
            arguments.append((deg,pol_i1))

        p_i = summer_funcs(arguments)

        return p_i

    
    # i,j are vector of indices of i1,...,id and j1,...,jd 
    # write it as a tensor train operator format where dimension is i_k x i_j
    # rank 1 for all core
    def Hk(self,indice_k):
        Hk = []
        for k in range(self.d):
            temp_Hk = np.zeros((1,self.pol_deg,self.pol_deg,1))
            psi_curr = self.psi[k]
            
            if k != (indice_k-1): # no need to take derivative for i then
                for j in range(self.pol_deg):
                    phi_curr_j = self.all_phi[k][j]
                    phi_curr_j = self.function_pi(phi_curr_j)
                    for i in range(self.pol_deg):
                        phi_curr_i = self.all_phi[k][i]
                        phi_curr_i = self.function_pi(phi_curr_i)
                        
                        func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi_curr(x)

                        int_result,err = integrate.quad(func,self.min_interval,self.max_interval)
                        temp_Hk[0,i,j,0] = self.P[k]*int_result

            if k == (indice_k -1):
                for j in range(self.pol_deg):
                    phi_curr_j = self.all_phi[k][j]
                    phi_curr_j = np.polyder(phi_curr_j)
                    phi_curr_j = self.function_pi(phi_curr_j)
                    for i in range(self.pol_deg):
                        phi_curr_i = self.all_phi[k][i]
                        phi_curr_i = np.polyder(phi_curr_i)
                        phi_curr_i = self.function_pi(phi_curr_i)

                        func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi_curr(x)

                        int_result,err = integrate.quad(func,self.min_interval,self.max_interval)
                        temp_Hk[0,i,j,0] = self.P[k]*int_result
            Hk.append(temp_Hk)
        return Hk
    
#     def TTOp_Hk(self,indice_k):
#         entries = self.Hk(indice_k)
#         Hk_tto = xe.TTOperator.random([self.pol_deg]*2*self.d,[1]*(self.d-1))

#         for i in range(self.d):
#             Hk_tto.set_component(i,xe.Tensor.from_ndarray(np.array(entries[i])))

#         return Hk_tto
    
    def HA(self):
        HA = []
        for k in range(self.d):
            temp_HA = np.zeros((1,self.pol_deg,self.pol_deg,1))
            psi_a_curr = self.psi_a[k]  # basis function a
            
            for j in range(self.pol_deg):
                phi_curr_j = self.all_phi[k][j]
                phi_curr_j = self.function_pi(phi_curr_j)
                for i in range(self.pol_deg):
                    phi_curr_i = self.all_phi[k][i]
                    phi_curr_i = self.function_pi(phi_curr_i)

                    func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi_a_curr(x)

                    int_result,err = integrate.quad(func,self.min_interval,self.max_interval)
                    temp_HA[0,i,j,0] = self.ak[k]*int_result
                    
            HA.append(temp_HA)
        return HA
    
#     def TTOp_HA(self):
#         entries = self.HA()
#         HA_tto = xe.TTOperator.random([self.pol_deg]*2*self.d,[1]*(self.d-1))

#         for i in range(self.d):
#             HA_tto.set_component(i,xe.Tensor.from_ndarray(np.array(entries[i])))

#         return HA_tto
    
    def HB(self):
        HB = []
        for k in range(self.d):
            temp_HB = np.zeros((1,self.pol_deg,self.pol_deg,1))
            psi_b_curr = self.psi_b[k]
            
            for j in range(self.pol_deg):
                phi_curr_j = self.all_phi[k][j]
                phi_curr_j = self.function_pi(phi_curr_j)
                for i in range(self.pol_deg):
                    phi_curr_i = self.all_phi[k][i]
                    phi_curr_i = self.function_pi(phi_curr_i)

                    func = lambda x: phi_curr_i(x)*phi_curr_j(x)*psi_b_curr(x)

                    int_result,err = integrate.quad(func,self.min_interval,self.max_interval)
                    temp_HB[0,i,j,0] = self.bk[k]*int_result
            HB.append(temp_HB)
        return HB
    
#     def TTOp_HB(self):
#         entries = self.HB()
#         HB_tto = xe.TTOperator.random([self.pol_deg]*2*self.d,[1]*(self.d-1))

#         for i in range(self.d):
#             HB_tto.set_component(i,xe.Tensor.from_ndarray(np.array(entries[i])))

#         return HB_tto

    def hb(self):
        hb = []
        for k in range(self.d):
            temp_hb = np.zeros((1,self.pol_deg,1))
            psi_b_curr = self.psi_b[k]
            
            for i in range(self.pol_deg):
                phi_curr_i = self.all_phi[k][i]
                phi_curr_i = self.function_pi(phi_curr_i)
                
                func = lambda x: phi_curr_i(x)*psi_b_curr(x)

                int_result,err = integrate.quad(func,self.min_interval,self.max_interval)
                temp_hb[0,i,0] = self.bk[k]*int_result
                
            hb.append(temp_hb)
        return hb


# +
# beta = 5
# d = 3
# max_pol_deg = 5
# comp = Component_easier(beta,d,max_pol_deg)
# HA = comp.HA()
# hb = comp.hb()
# HB = comp.HB()
# print(HA[0])

# +
# print(HB[0])

# +
# print(hb[0])
# -



