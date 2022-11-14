# + active=""
# """
# Store the function psi^{k},a^{k},b^{k} for each dimension k (only one function to be returned)
#
# For the double well potential function
# """
# -


from scipy import integrate
import numpy as np
from numpy import exp
from scipy import special
from math import pi
import orth_pol_q
from scipy import interpolate
import scipy


# +
class orth_pab:
    def __init__(self,beta,d,max_pol_deg,max_pol_deg_px):
        self.beta = beta
        self.d = d
        self.max_pol_deg = max_pol_deg
        self.pol_deg = max_pol_deg+1
        self.pol_deg_px = max_pol_deg_px+1
        
        
        self.a1 = -1.5
        self.a2 = 1
        self.b1 = -0.5
        self.b2 = 2
        
        
        self.a = -1
        self.b = 1
        
        self.max_pol_deg_vec = [max_pol_deg]*d
        self.max_pol_deg_vec_px = [max_pol_deg_px]*2 + [max_pol_deg]*(d-2)
        self.all_poly = orth_pol_q.all_pol_basis(self.max_pol_deg_vec,d,beta) #basis functions for committor function
        self.px_poly = orth_pol_q.all_pol_px(self.max_pol_deg_vec_px,d,beta) # first two functions are lambda functions 
        # for other dimensions, poly1d
        
    def orth_p(self,dim):
        """
        we skip the case when dim = 0(1st dimension) since the coefficients for the first two dimensions
        has to be generated together
        k = dim is the actual dimension-1
        """
        if dim == 1:
            poly = self.px_poly[0:2]
            coef = self.coef_px_2D() 
        else: 
            x_vec = np.linspace(self.a,self.b,self.pol_deg+1) 
            func_lbd = lambda x: exp(-self.beta*(x**2)/(2*0.05**2))
            
            y = func_lbd(x_vec)
            coef = np.polynomial.legendre.legfit(x_vec,y,self.max_pol_deg) # coefficients in standard polynomials
            poly = self.px_poly[dim]

            zk_temp, err = integrate.quad(func_lbd,self.a,self.b)
#             coef = coef/zk_temp


        return poly,coef

    def orth_ab(self,RAB,xAB):
        # xA is a list 
        sigma = RAB/np.sqrt(len(xAB))
        orth = []
        abk = []
        sizeAB = len(xAB)
        for i in range(self.d):
            if i < (sizeAB):
                f = lambda x,y=xAB[i]: exp(-((x - y)**2)/(2*sigma**2))
                abk_temp = 1/(sigma*(2*pi)**0.5)
            else: 
                f = lambda x: 1
                abk_temp = 1
            
            orth.append(f)
            abk.append(abk_temp)
            
        return orth,abk
    
#     def orth_ab(self,RAB,xAB):
#         # xA is a list 
#         sigma = RAB/np.sqrt(len(xAB))
#         orth = []
#         abk = []
#         sizeAB = len(xAB)
#         for i in range(self.d):
#             if i == 0:
#                 f = lambda x: exp(-((x - xAB[0])**2)/(2*sigma**2))
#                 abk_temp = 1/(sigma*(2*pi)**0.5)
#             elif i == 1:
#                 f = lambda x: exp(-((x - xAB[1])**2)/(2*sigma**2))
#                 abk_temp = 1/(sigma*(2*pi)**0.5)
#             else: 
#                 f = lambda x: 1
#                 abk_temp = 1
            
#             orth.append(f)
#             abk.append(abk_temp)
            
#         return orth,abk
    
    
    def coef_px_2D(self):
        n = self.pol_deg_px # the number of Chebyshev nodes
        i = np.array(range(n-1,-1,-1))
        t = pi*(0.5+i)/n
        z = np.cos(t)
        # % Chebyshev grid shifted to [xmin,xmax]*[ymin,ymax]
        xt,yt = np.meshgrid(0.5*(self.a1+self.a2 + z*(self.a2-self.a1)),0.5*(self.b1+self.b2+z*(self.b2-self.b1)))
        Z_2D_= self.Z_2D()
        f = self.funU(xt,yt,Z_2D_)
        cosktj = np.cos(np.array(range(n))[:,None]*t)
        coefx = np.zeros((n,n))
        # % interpolate f along each grid line y = const

        for r in range(n):
            if r == 0:
                fp = f[r,:][:,None]
                coefx[r,:] = 2*(cosktj@fp).T/n
            else:
                fp = f[r,:][:,None]
                coefx[r,:] = 2*(cosktj@fp).T/n

        # % interpolate each coefficient coefx(:,j) by a Cheb sum
        coefxy = np.zeros((n,n))
        for r in range(n):
            if r == 0:
                coefxy[r,:] = 2*cosktj[r,:]@coefx/n
            else:
                coefxy[r,:] = 2*cosktj[r,:]@coefx/n
        
        coefxy[0,:] = 0.5*coefxy[0,:]
        coefxy[:,0] = 0.5*coefxy[:,0]
        coefxy = coefxy.T
        
        u, s, vh = np.linalg.svd(coefxy)
        P1 = u*s
        P2 = vh
        
        return P1,P2 
    
    def Z_2D(self):
        a = np.array([-1,-1,-6.5,0.7])
        b = np.array([0,0,11,0.6])
        c = np.array([-10,-10,-6.5,0.7])
        D = np.array([-200,-100,-170,15])
        X = np.array([1,0,-0.5,-1])
        Y = np.array([0,0.5,1.5,1])
        gamma = 9
        k = 5

        fx1 = lambda y,x: D[0]*np.exp(a[0]*((x-X[0])**2) + b[0]*(x-X[0])*(y-Y[0]) + c[0]*(y**2))
        fx2 = lambda y,x: D[1]*np.exp(a[1]*((x-X[1])**2) + b[1]*(x-X[1])*(y-Y[1]) + c[1]*((y-Y[1])**2))
        fx3 = lambda y,x: D[2]*np.exp(a[2]*((x-X[2])**2) + b[2]*(x-X[2])*(y-Y[2]) + c[2]*((y-Y[2])**2))
        fx4 = lambda y,x: D[3]*np.exp(a[3]*((x-X[3])**2) + b[3]*(x-X[3])*(y-Y[3]) + c[3]*((y-Y[3])**2))
        fx = lambda y, x: np.exp(-self.beta * (fx1(y,x) + fx2(y,x) + fx3(y,x) + fx4(y,x)))

        Z,error = scipy.integrate.dblquad(fx, self.a1,self.a2, lambda x: self.b1, lambda x: self.b2, epsabs=1.49e-08, epsrel=1.49e-08)
        
        return Z
    
    def funU(self,x,y,Z):
        a = np.array([-1,-1,-6.5,0.7])
        b = np.array([0,0,11,0.6])
        c = np.array([-10,-10,-6.5,0.7])
        D = np.array([-200,-100,-170,15])
        X = np.array([1,0,-0.5,-1])
        Y = np.array([0,0.5,1.5,1])
        gamma = 9
        k = 5

        fx1 = D[0]*np.exp(a[0]*((x-X[0])**2) + b[0]*(x-X[0])*(y-Y[0]) + c[0]*(y**2))
        fx2 = D[1]*np.exp(a[1]*((x-X[1])**2) + b[1]*(x-X[1])*(y-Y[1]) + c[1]*((y-Y[1])**2))
        fx3 = D[2]*np.exp(a[2]*((x-X[2])**2) + b[2]*(x-X[2])*(y-Y[2]) + c[2]*((y-Y[2])**2))
        fx4 = D[3]*np.exp(a[3]*((x-X[3])**2) + b[3]*(x-X[3])*(y-Y[3]) + c[3]*((y-Y[3])**2))
#         extra = gamma*np.sin(2*k*pi*x)*np.sin(2*k*pi*y)

        U = fx1+fx2+fx3+fx4
    #     U = U+extra
        U = np.exp(-self.beta*U)/Z

        return U
    


# +
# # set_A = [-0.558,1.441]
# set_A = np.array([0.623,0.028])
# print(set_A[1])
# RA = 0.1
# beta = 1/30
# d = 3
# max_pol_deg = 5
# pab = orth_pab(beta,d,max_pol_deg)
# B,C = pab.orth_ab(RA,set_A)
# x = 0.623
# print('difference',B[1](x) - B[0](x))
# print('B[0]: ', B[0](x))


# sigma = RA/np.sqrt(2)
# f = lambda x: exp(-((x - set_A[0])**2)/(2*sigma**2))
# print(f(-0.558))
# +
# #beta,d,max_pol_deg,max_pol_deg_px
# pab = orth_pab(1/20,2,25,109)
# psi_a,ak = pab.orth_p(1)
# # psi_b,bk = pab.orth_ab(sigma=0.05,c=1)

# print(np.shape(ak))

# +
# beta = 1/30
# Z_2D(beta)
# -


class orth_pab_rugged:
    def __init__(self,beta,d,max_pol_deg,max_pol_deg_px):
        self.beta = beta
        self.d = d
        self.max_pol_deg = max_pol_deg
        self.pol_deg = max_pol_deg+1
        self.pol_deg_px = max_pol_deg_px+1
        
        
        self.a1 = -1.5
        self.a2 = 1
        self.b1 = -0.5
        self.b2 = 2
        
        
        self.a = -1
        self.b = 1
        
        self.max_pol_deg_vec = [max_pol_deg]*d
        self.max_pol_deg_vec_px = [max_pol_deg_px]*2 + [max_pol_deg]*(d-2)
        self.all_poly = orth_pol_q.all_pol_basis(self.max_pol_deg_vec,d,beta) #basis functions for committor function
        self.px_poly = orth_pol_q.all_pol_px(self.max_pol_deg_vec_px,d,beta) # first two functions are lambda functions 
        # for other dimensions, poly1d
        
    def orth_p(self,dim):
        """
        we skip the case when dim = 0(1st dimension) since the coefficients for the first two dimensions
        has to be generated together
        k = dim is the actual dimension-1
        """
        if dim == 1:
            poly = self.px_poly[0:2]
            coef = self.coef_px_2D() 
        else: 
            x_vec = np.linspace(self.a,self.b,self.pol_deg+1) 
            func_lbd = lambda x: exp(-self.beta*(x**2)/(2*0.05**2))
            
            y = func_lbd(x_vec)
            coef = np.polynomial.legendre.legfit(x_vec,y,self.max_pol_deg) # coefficients in standard polynomials
            poly = self.px_poly[dim]

            zk_temp, err = integrate.quad(func_lbd,self.a,self.b)
#             coef = coef/zk_temp


        return poly,coef

    def orth_ab(self,RAB,xAB):
        # xA is a list 
        sigma = RAB/np.sqrt(len(xAB))
        orth = []
        abk = []
        sizeAB = len(xAB)
        for i in range(self.d):
            if i < (sizeAB):
                f = lambda x,y=xAB[i]: exp(-((x - y)**2)/(2*sigma**2))
                abk_temp = 1/(sigma*(2*pi)**0.5)
            else: 
                f = lambda x: 1
                abk_temp = 1
            
            orth.append(f)
            abk.append(abk_temp)
            
        return orth,abk
    
    
    def coef_px_2D(self):
        n = self.pol_deg_px # the number of Chebyshev nodes
        i = np.array(range(n-1,-1,-1))
        t = pi*(0.5+i)/n
        z = np.cos(t)
        # % Chebyshev grid shifted to [xmin,xmax]*[ymin,ymax]
        xt,yt = np.meshgrid(0.5*(self.a1+self.a2 + z*(self.a2-self.a1)),0.5*(self.b1+self.b2+z*(self.b2-self.b1)))
        Z_2D_= self.Z_2D()
        f = self.funU_rugged(xt,yt,Z_2D_)
        cosktj = np.cos(np.array(range(n))[:,None]*t)
        coefx = np.zeros((n,n))
        # % interpolate f along each grid line y = const

        for r in range(n):
            if r == 0:
                fp = f[r,:][:,None]
                coefx[r,:] = 2*(cosktj@fp).T/n
            else:
                fp = f[r,:][:,None]
                coefx[r,:] = 2*(cosktj@fp).T/n

        # % interpolate each coefficient coefx(:,j) by a Cheb sum
        coefxy = np.zeros((n,n))
        for r in range(n):
            if r == 0:
                coefxy[r,:] = 2*cosktj[r,:]@coefx/n
            else:
                coefxy[r,:] = 2*cosktj[r,:]@coefx/n
        
        coefxy[0,:] = 0.5*coefxy[0,:]
        coefxy[:,0] = 0.5*coefxy[:,0]
        coefxy = coefxy.T
        
        u, s, vh = np.linalg.svd(coefxy)
        P1 = u*s
        P2 = vh
        
        return P1,P2 
    
    def Z_2D(self):
        a = np.array([-1,-1,-6.5,0.7])
        b = np.array([0,0,11,0.6])
        c = np.array([-10,-10,-6.5,0.7])
        D = np.array([-200,-100,-170,15])
        X = np.array([1,0,-0.5,-1])
        Y = np.array([0,0.5,1.5,1])
        gamma = 9
        k = 5

        fx1 = lambda y,x: D[0]*np.exp(a[0]*((x-X[0])**2) + b[0]*(x-X[0])*(y-Y[0]) + c[0]*(y**2))
        fx2 = lambda y,x: D[1]*np.exp(a[1]*((x-X[1])**2) + b[1]*(x-X[1])*(y-Y[1]) + c[1]*((y-Y[1])**2))
        fx3 = lambda y,x: D[2]*np.exp(a[2]*((x-X[2])**2) + b[2]*(x-X[2])*(y-Y[2]) + c[2]*((y-Y[2])**2))
        fx4 = lambda y,x: D[3]*np.exp(a[3]*((x-X[3])**2) + b[3]*(x-X[3])*(y-Y[3]) + c[3]*((y-Y[3])**2))
        extra = lambda y,x: gamma*np.sin(2*k*pi*x)*np.sin(2*k*pi*y)
        fx = lambda y, x: np.exp(-self.beta * (fx1(y,x) + fx2(y,x) + fx3(y,x) + fx4(y,x) + extra(x,y)))

        Z,error = scipy.integrate.dblquad(fx, self.a1,self.a2, lambda x: self.b1, lambda x: self.b2, epsabs=1.49e-08, epsrel=1.49e-08)
        
        return Z
    
    def funU_rugged(self,x,y,Z):
        a = np.array([-1,-1,-6.5,0.7])
        b = np.array([0,0,11,0.6])
        c = np.array([-10,-10,-6.5,0.7])
        D = np.array([-200,-100,-170,15])
        X = np.array([1,0,-0.5,-1])
        Y = np.array([0,0.5,1.5,1])
        gamma = 9
        k = 5
        
        extra = gamma*np.sin(2*k*pi*x)*np.sin(2*k*pi*y)
        fx1 = D[0]*np.exp(a[0]*((x-X[0])**2) + b[0]*(x-X[0])*(y-Y[0]) + c[0]*(y**2))
        fx2 = D[1]*np.exp(a[1]*((x-X[1])**2) + b[1]*(x-X[1])*(y-Y[1]) + c[1]*((y-Y[1])**2))
        fx3 = D[2]*np.exp(a[2]*((x-X[2])**2) + b[2]*(x-X[2])*(y-Y[2]) + c[2]*((y-Y[2])**2))
        fx4 = D[3]*np.exp(a[3]*((x-X[3])**2) + b[3]*(x-X[3])*(y-Y[3]) + c[3]*((y-Y[3])**2))
#         extra = gamma*np.sin(2*k*pi*x)*np.sin(2*k*pi*y)

        U = fx1+fx2+fx3+fx4
    #     U = U+extra
        U = np.exp(-self.beta*U)/Z

        return U






# +
class orth_pab_easier:
    def __init__(self,beta,d,max_pol_deg):
        self.beta = beta
        self.d = d
        self.max_pol_deg = max_pol_deg
        self.pol_deg = max_pol_deg+1
        self.a = -1
        self.b = 1
        self.max_pol_deg_vec = [max_pol_deg]*d
        self.all_poly = orth_pol_q.all_pol(self.max_pol_deg_vec,d,beta,self.a,self.b)
        
    def orth_p(self):
        orth = []
        coef = []
        
        for i in range(self.d):
            if i == 0:
                func_lbd = lambda x: exp(-self.beta*(x**2 - 1)**2)
                zk_temp, err = integrate.quad(func_lbd,self.a,self.b)
            else:
                func_lbd = lambda x: exp(-self.beta*0.3*(x**2))
                zk_temp, err = integrate.quad(func_lbd,self.a,self.b)
            
            orth.append(func_lbd)
            coef.append(zk_temp)

        return orth,coef
    
#     def orth_ab(self,dim,sigma,c):
#         x_vec = np.linspace(self.a,self.b,self.pol_deg+1) 
#         if dim == 0:
#             func_lbd = lambda x: 1/(sigma*(2*pi)**0.5)*exp(-((x-c)**2)/(2*sigma))
#             y = func_lbd(x_vec)
#         else:
#             y = np.ones(self.pol_deg+1)
            
#         orth = self.all_poly[dim]
#         coef = np.polynomial.legendre.legfit(x_vec,y,self.max_pol_deg)
            
#         return orth,coef
    
#    # functions for orthogonal basis a or b
## easier version
    def orth_ab(self,sigma,c):
        orth = []
        abk = []
        for i in range(self.d):
            if i == 0:
                f = lambda x: exp(-(x - c)**2/(2*sigma**2))
                abk_temp = 1/(sigma*(2*pi)**0.5)
            else:
                f = lambda x: 1
                abk_temp = 1
            
            orth.append(f)
            abk.append(abk_temp)
            
        return orth,abk

# -





