"""
Calcululates orthonormal polynomials of order min_pol_deg to max_pol_deg on the interval (a, b) with weight 
function given by the equilibrium distribution e^-beta V(x) where V(x) is the potential function

Specifically for the double well potential

compute basis function for committor function q(x)
"""


from scipy import integrate
import numpy as np
from numpy import exp
from scipy import special
import math

"""
three-term relation for orthogonal polynomials
"""

# +
# # a: lower bound of the interval
# # b: upper bound of the interval 
# # maximum polynomial degree
# # calculate the polynomials for the first dimension k = 1
# def calc_pol_d1(beta,max_pol_deg, min_pol_deg=0,a=-1,b=1):
#     weight_1 = lambda x:exp(-beta*(x**2-1)**2)
    
#     # evaluate the inner product
#     def eval_inner_product(p1,p2,weight,a=-1,b=1):
#         f = lambda x: p1(x)*p2(x)*weight(x)
#         inner_product,error = integrate.quad(f,a,b)
#         return inner_product


#     # this turn the poly1d into a symbolic function for integration
#     def function_pi(poly_k):
#         # k goes from 0 to j-1 in the for loop 
#         def fun_pi(i1,pol_i1):
#             return lambda x: (x**i1)*pol_i1

#         def summer_funcs(args):
#             return lambda x: sum(fun_pi(*a)(x) for a in args)

#         arguments = []
#         for i1 in range(len(poly_k)):
#             deg = len(poly_k)-1 - i1
#             pol_i1 = poly_k[i1]
#             arguments.append((deg,pol_i1))

#         p_i = summer_funcs(arguments)
#         return p_i
    
#     pol_deg = max_pol_deg + 1
    
#     ## Initialization
#     pol = []
#     pol.append(np.poly1d([1]))
    
    
#     ## perform the three term recurrence relation
#     for j in range(1,pol_deg):
#         if j != 1:  
#             p_pre = pol[j-1] #p_n
#             p_prepre = pol[j-2] #p_{n-1}
#         else:
#             p_pre = pol[j-1]
#             p_prepre = np.poly1d([0])
        
        
#         x_pn = p_pre.c
#         x_pn = np.append(x_pn,0)
#         x_pn = np.poly1d(x_pn) # poly1d of x*p_n
        
#         p_pre_lbd = function_pi(p_pre.c)
#         p_prepre_lbd = function_pi(p_prepre.c) # turn the p_{n-1} and p_{n} into lambda functions for inner product
#         x_pn_lbd = function_pi(x_pn.c)
        
#         alpha = eval_inner_product(p_pre_lbd,x_pn_lbd,weight_1)/eval_inner_product(p_pre_lbd,p_pre_lbd,weight_1)
        
#         if j != 1:
#             beta = eval_inner_product(p_pre_lbd,p_pre_lbd,weight_1)/eval_inner_product(p_prepre_lbd,p_prepre_lbd,weight_1)
#         else: 
#             beta = 0
            
#         p_n = x_pn - alpha*p_pre - beta*p_prepre
#         pol.append(p_n)
        
#     for i0 in range(len(pol)):
#         pol[i0] /= pol[0](0)
#     #pol[0][0] = 1
    
#     return pol

# +
# # a: lower bound of the interval
# # b: upper bound of the interval 
# # maximum polynomial degree
# # calculate the polynomials for the first dimension k = 1
# def calc_pol_d2(beta,max_pol_deg, min_pol_deg=0,a=-1,b=1):
#     weight = lambda x:exp(-beta*0.3*x**2)
    
#     # evaluate the inner product
#     def eval_inner_product(p1,p2,weight,a = -1,b = 1):
#         f = lambda x: p1(x)*p2(x)*weight(x)
#         inner_product,error = integrate.quad(f,a,b)
#         return inner_product


#     # this turn the poly1d into a symbolic function for integration
#     def function_pi(poly_k):
#         # k goes from 0 to j-1 in the for loop 
#         def fun_pi(i1,pol_i1):
#             return lambda x: (x**i1)*pol_i1

#         def summer_funcs(args):
#             return lambda x: sum(fun_pi(*a)(x) for a in args)

#         arguments = []
#         for i1 in range(len(poly_k)):
#             deg = len(poly_k)-1 - i1
#             pol_i1 = poly_k[i1]
#             arguments.append((deg,pol_i1))

#         p_i = summer_funcs(arguments)
#         return p_i
    
#     pol_deg = max_pol_deg + 1
    
#     ## Initialization
#     pol = []
#     pol.append(np.poly1d([1]))
    
    
#     ## perform the three term recurrence relation
#     for j in range(1,pol_deg):
#         if j != 1:  
#             p_pre = pol[j-1] #p_n
#             p_prepre = pol[j-2] #p_{n-1}
#         else:
#             p_pre = pol[j-1]
#             p_prepre = np.poly1d([0])
        
        
#         x_pn = p_pre.c
#         x_pn = np.append(x_pn,0)
#         x_pn = np.poly1d(x_pn) # poly1d of x*p_n
        
#         p_pre_lbd = function_pi(p_pre.c)
#         p_prepre_lbd = function_pi(p_prepre.c) # turn the p_{n-1} and p_{n} into lambda functions for inner product
#         x_pn_lbd = function_pi(x_pn.c)
        
#         alpha = eval_inner_product(p_pre_lbd,x_pn_lbd,weight)/eval_inner_product(p_pre_lbd,p_pre_lbd,weight)
        
#         if j != 1:
#             beta = eval_inner_product(p_pre_lbd,p_pre_lbd,weight)/eval_inner_product(p_prepre_lbd,p_prepre_lbd,weight)
#         else: 
#             beta = 0
            
#         p_n = x_pn - alpha*p_pre - beta*p_prepre
#         pol.append(p_n)
        
#     for i0 in range(len(pol)):
#         pol[i0] /= pol[0](0)
#     #pol[0][0] = 1
    
#     return pol
# -

def calc_pol(max_pol_deg):
    pol = []
    pol_deg = max_pol_deg+1
    for deg in range(pol_deg):
        p_deg = special.legendre(deg)
        pol.append(p_deg)
        
    return pol


# d is the dimension
# max_pol_deg_vec(1 x d) is a vector containing degree of polynomials for each dimension
# all_pol[i][j] is the basis function of the (i+1)th dimension and (j+1)th polynomials (degree j)
def all_pol_basis(max_pol_deg_vec,d,beta):
    all_pol_M = []
    for i in range(d):
        max_pol_deg = max_pol_deg_vec[i]
        pol = calc_pol(max_pol_deg)
        
        all_pol_M.append(pol)
        
    return all_pol_M


def calc_pol_chebyshev(max_pol_deg):
    pol = []
    pol_deg = max_pol_deg+1
    coef = [1]
    for deg in range(pol_deg):
        p_deg = np.polynomial.chebyshev.Chebyshev(coef)
        pol.append(p_deg)
        coef.insert(0,0)
    
    return pol


def all_pol_px(max_pol_deg_vec,d,beta):
    all_pol_M = []
    for i in range(d):
        max_pol_deg = max_pol_deg_vec[i]
        if i in [0,1]:
            pol = calc_pol_chebyshev(max_pol_deg)
        else: 
            pol = calc_pol(max_pol_deg)
            
        all_pol_M.append(pol)
        
    return all_pol_M


"""
Gram-schmidt
"""

# +
# # a: lower bound of the interval
# # b: upper bound of the interval 
# # maximum polynomial degree
# # calculate the polynomials for the first dimension k = 1
# def calc_pol_d1(a,b,beta,max_pol_deg, min_pol_deg=0):
#     weight_1 = lambda x:exp(-beta*(x**2-1)**2)
    
#     # evaluate the inner product
#     def eval_inner_product(a,b,p1,p2,weight):
#         f = lambda x: p1(x)*p2(x)*weight(x)
#         inner_product,error = integrate.quad(f,a,b)
#         return inner_product


#     # this turn the poly1d into a symbolic function for integration
#     def function_pi(poly_k):
#         # k goes from 0 to j-1 in the for loop 
#         def fun_pi(i1,pol_i1):
#             return lambda x: (x**i1)*pol_i1

#         def summer_funcs(args):
#             return lambda x: sum(fun_pi(*a)(x) for a in args)

#         arguments = []
#         for i1 in range(len(poly_k)):
#             deg = len(poly_k)-1 - i1
#             pol_i1 = poly_k[i1]
#             arguments.append((deg,pol_i1))

#         p_i = summer_funcs(arguments)
#         return p_i
    
    
#     ## initialization
#     pol_deg = max_pol_deg + 1
#     polynomials = []
#     for i0 in range(0, pol_deg):
#         polynomials.append([1])
    
    
#     for i0 in range(0,pol_deg):
#         for i1 in range(min_pol_deg):
#             polynomials[i0].append(0)
#     #    polynomials[i0].append(0)
#         for i1 in range(0, i0):
#             polynomials[i0].append(0)
    
#     pol = []
    
#     for i0 in range(pol_deg):
#         pol.append(np.poly1d(polynomials[i0]))
        
        
#     ## perform gram-schmidt to obtain the basis functions of polynomials 
    
#     for j in range(pol_deg):
#         temp_pol = 1*pol[j]
#         p_j = function_pi(polynomials[j])
#         for k in range(j):
#             p_k = function_pi(polynomials[k])
#             temp_pol = temp_pol - eval_inner_product(a,b,p_j,p_k,weight_1)/eval_inner_product(a,b,p_k,p_k,weight_1)* pol[i1] 
#         temp_pol_lbd = function_pi(temp_pol.c)
#         temp_pol = temp_pol / np.sqrt(eval_inner_product(a,b,temp_pol_lbd,temp_pol_lbd,weight_1))
#         pol[j] = 1* temp_pol
#     for i0 in range(len(pol)):
#         ## len(pol) = number of polynomials stored
#         ## normalize the first polynomial
#         pol[i0] /= pol[0](0)
#     #pol[0][0] = 1
    
#     return pol

# +
# def calc_pol_d2(max_pol_deg):
#     pol_d2 = []
#     pol_deg = max_pol_deg+1
#     for deg in range(pol_deg):
#         p_deg = special.hermite(deg,monic=True)
#         pol_d2.append(p_deg)
        
#     return pol_d2
# -


