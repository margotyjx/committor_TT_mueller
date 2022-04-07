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

"""
Testing points in 2D and plot the estimation by the computed result for T = 30
"""

# +
## result from finite difference

# +
# posf = pd.DataFrame(new_pos,columns=['X','Y']) #convert to a dataframe

# posf.to_excel('xy_TT_mueller.xlsx',index=False) #save to file

# +
import pandas as pd
FD_file = pd.read_excel('FEMdata_T100_uniform.xlsx')

points_x = pd.DataFrame(FD_file['X'])
points_y = pd.DataFrame(FD_file['Y'])
target = pd.DataFrame(FD_file['TARGET'])

pos = np.hstack((np.array(points_x.values),np.array(points_y.values)))
target_np = np.array(target.values)

# +
new_pos = []
new_target = []
for i in range(len(points_x)):
    if pos[i,0] >= -1.5 and pos[i,0]<= 1 and pos[i,1] >= -0.5 and pos[i,1]<=2:
        new_pos.append(pos[i,:])
        new_target.append(target_np[i])
        
new_pos = np.array(new_pos)
print(new_pos.shape)
# -

new_pos = np.array(new_pos)
new_target = np.array(new_target)

# +
Test_data = np.hstack((new_pos,new_target))
Test_data_dataframe= pd.DataFrame(Test_data,columns=['X','Y','committor']) #convert to a dataframe

Test_data_dataframe.to_excel('test_data.xlsx',index=False) #save to file

# +
plt.scatter(new_pos[:,0],new_pos[:,1],c = new_target, s = 20)
plt.colorbar()
plt.title('approximation by FEM')

# plt.scatter(new_pos[:,0],eval_result,color = 'C1',label = 'q_TT',s = 10)
# plt.legend()
# plt.xlabel('x1')
# plt.ylabel('committor function')
# plt.savefig('2D_beta5.png')
# plt.show()
# -



# +
beta = 1/100
a1 = -1.5
a2 = 1
b1 = -0.5
b2 = 2
new_pos_trans = np.copy(new_pos)
new_pos_trans = np.copy(new_pos)
new_pos_trans[:,0] =(2*new_pos_trans[:,0] - (a1 + a2))/(a2 - a1)
new_pos_trans[:,1] =(2*new_pos_trans[:,1] - (b1 + b2))/(b2 - b1)

eval_result = eval_Q(beta,new_pos_trans,"Q")
plt.scatter(new_pos[:,0],new_pos[:,1],c= eval_result,s = 20)
plt.colorbar()
plt.title('approximation by TT')
# -

err_vec = new_target-eval_result[:,None]
plt.scatter(new_pos[:,0],new_pos[:,1],c = err_vec)
plt.title('error: q_true - q')
plt.colorbar()

# relative error
norm2_err = np.linalg.norm(err_vec,2)
norm2_target = np.linalg.norm(target_np,2)
relative_error_2D = norm2_err/norm2_target
print(relative_error_2D)

# RMSE
square_errer_2D = np.square(err_vec)
RMSE_2D = np.sqrt(square_errer_2D.mean())
print(RMSE_2D)



"""
Testing points in 2D, beta = 20 and plot the estimation by the computed result
"""

x = np.linspace(-1,1,101)
y = np.linspace(-1,1,101)
YY,XX = np.meshgrid(x,y)
positions2 = np.vstack((XX.ravel(),YY.ravel()))
new_pos = np.transpose(positions2)
print(new_pos.shape)

# +
## result from finite difference

# +
beta = 20
eval_result = eval_Q(beta,new_pos,"Q")
import matplotlib.pyplot as plt
import pandas as pd
FD_file = pd.read_excel('FD_beta20.xlsx')

points = pd.DataFrame(FD_file['X'])
target = pd.DataFrame(FD_file['Y'])
plt.plot(points,target,label = 'q_true')
plt.scatter(new_pos[:,0],eval_result,color = 'C1',label = 'q_TT',s = 10)
plt.legend()
plt.xlabel('x1')
plt.ylabel('committor function')
plt.show()
# -

target_np = np.array(target.values)
target_new = np.repeat(target_np,41)

x = np.linspace(-1,1,41)
y = np.linspace(-1,1,41)
YY,XX = np.meshgrid(x,y)
positions2 = np.vstack((XX.ravel(),YY.ravel()))
new_pos = np.transpose(positions2)
beta = 20
eval_result = eval_Q(beta,new_pos,'Q')

err_vec = target_new-eval_result
print(err_vec.shape)

plt.plot(new_pos[:,0],err_vec)
plt.title('error: q_true - q')
plt.xlabel('x1')
plt.ylabel('Residual')

# relative error
norm2_err = np.linalg.norm(err_vec,2)
norm2_target = np.linalg.norm(target_np,2)
relative_error_2D = norm2_err/norm2_target
print(relative_error_2D)

# MSE
square_errer_2D = np.square(err_vec)
RMSE_2D = np.sqrt(square_errer_2D.mean())
print(RMSE_2D)

"""
Testing points in 3D, beta = 5 and plot the estimation by the computed result
"""

D = 6
x = np.linspace(-1,1,D)
x2 = np.linspace(-1,1,41)
x3 = np.linspace(-1,1,D)
x4 = np.linspace(-1,1,D)
x5 = np.linspace(-1,1,D)
x6 = np.linspace(-1,1,D)
x7 = np.linspace(-1,1,D)
x8 = np.linspace(-1,1,D)
x9 = np.linspace(-1,1,D)
x10 = np.linspace(-1,1,D)
YY,XX,ZZ = np.meshgrid(*[x,x2,x3])
# print('YY',YY)
# print('XX',XX)
# print('ZZ',ZZ)
positions2 = np.vstack((XX.ravel(),YY.ravel(),ZZ.ravel()))
print(positions2.shape)
new_pos = np.transpose(positions2)
print(new_pos.shape)

beta = 20
eval_result = eval_Q(beta,new_pos,"Q_3D_beta5")

# +
# print(eval_result)
# -

import matplotlib.pyplot as plt
plt.scatter(new_pos[:,0],eval_result)

# +
## result from finite difference

# +
import pandas as pd
FD_file = pd.read_excel('FD.xlsx')

points = pd.DataFrame(FD_file['X'])
target = pd.DataFrame(FD_file['Y'])
plt.plot(points,target,label = 'q_true')
plt.scatter(new_pos[:,0],eval_result,color = 'C1',label = 'q_TT',s = 10)
plt.legend()
plt.xlabel('x1')
plt.ylabel('committor function')
plt.show()
# -

target_np = np.array(target.values)
target_new = np.repeat(target_np,36)

# +
# x = np.linspace(-1,1,41)
# y = np.linspace(-1,1,41)
# YY,XX = np.meshgrid(x,y)
# positions2 = np.vstack((XX.ravel(),YY.ravel()))
# new_pos = np.transpose(positions2)
# beta = 20
# eval_result = eval_Q(beta,new_pos,'Q_3D_beta5')
# -

err_vec = target_new-eval_result
print(err_vec.shape)

plt.plot(new_pos[:,0],err_vec)
plt.title('error: q_true - q')
plt.xlabel('x1')
plt.ylabel('Residual')

# relative error
norm2_err = np.linalg.norm(err_vec,2)
norm2_target = np.linalg.norm(target_np,2)
relative_error_2D = norm2_err/norm2_target
print(relative_error_2D)

# MSE
square_errer_2D = np.square(err_vec)
RMSE_2D = np.sqrt(square_errer_2D.mean())
print(RMSE_2D)

"""
Testing points in 4D and plot the estimation by the computed result
"""

D = 6
x = np.linspace(-1,1,D)
x2 = np.linspace(-1,1,41)
x3 = np.linspace(-1,1,D)
x4 = np.linspace(-1,1,D)
x5 = np.linspace(-1,1,D)
x6 = np.linspace(-1,1,D)
x7 = np.linspace(-1,1,D)
x8 = np.linspace(-1,1,D)
x9 = np.linspace(-1,1,D)
x10 = np.linspace(-1,1,D)
YY,XX,ZZ,X4 = np.meshgrid(*[x,x2,x3,x4])
# print('YY',YY)
# print('XX',XX)
# print('ZZ',ZZ)
positions2 = np.vstack((XX.ravel(),YY.ravel(),ZZ.ravel(),X4.ravel()))
print(positions2.shape)
new_pos = np.transpose(positions2)
print(new_pos.shape)

beta = 5
eval_result = eval_Q(beta,new_pos,'Q')
import matplotlib.pyplot as plt
plt.scatter(new_pos[:,0],eval_result)

# +
# print(eval_result)
# -

"""
Result using finite difference method
"""

# +
import pandas as pd
FD_file = pd.read_excel('FD.xlsx')

points = pd.DataFrame(FD_file['X'])
target = pd.DataFrame(FD_file['Y'])
plt.plot(points,target,label = 'q_true')
plt.scatter(new_pos[:,0],eval_result,color = 'C1',label = 'q_TT',s = 10)
plt.legend()
plt.xlabel('x1')
plt.ylabel('committor function')
plt.show()
# -

target_np = np.array(target.values)
target_new = np.repeat(target_np,216)

# +
# x = np.linspace(-1,1,41)
# y = np.linspace(-1,1,41)
# YY,XX = np.meshgrid(x,y)
# positions2 = np.vstack((XX.ravel(),YY.ravel()))
# new_pos = np.transpose(positions2)
# beta = 20
# eval_result = eval_Q(beta,new_pos)
# -

err_vec = target_new-eval_result
print(err_vec.shape)

plt.plot(new_pos[:,0],err_vec)
plt.title('error: q_true - q')
plt.xlabel('x1')
plt.ylabel('Residual')

# relative error
norm2_err = np.linalg.norm(err_vec,2)
norm2_target = np.linalg.norm(target_np,2)
relative_error_4D = norm2_err/norm2_target
print(relative_error_4D)

# MSE
square_errer_4D = np.square(err_vec)
RMSE_4D = np.sqrt(square_errer_4D.mean())
print(RMSE_4D)

"""
Testing points in 4D, beta = 20, and plot the estimation by the computed result
"""

D = 6
x = np.linspace(-1,1,D)
x2 = np.linspace(-1,1,41)
x3 = np.linspace(-1,1,D)
x4 = np.linspace(-1,1,D)
x5 = np.linspace(-1,1,D)
x6 = np.linspace(-1,1,D)
x7 = np.linspace(-1,1,D)
x8 = np.linspace(-1,1,D)
x9 = np.linspace(-1,1,D)
x10 = np.linspace(-1,1,D)
YY,XX,ZZ,X4 = np.meshgrid(*[x,x2,x3,x4])
# print('YY',YY)
# print('XX',XX)
# print('ZZ',ZZ)
positions2 = np.vstack((XX.ravel(),YY.ravel(),ZZ.ravel(),X4.ravel()))
print(positions2.shape)
new_pos = np.transpose(positions2)
print(new_pos.shape)

beta = 20
eval_result = eval_Q(beta,new_pos,'Q')
import matplotlib.pyplot as plt
plt.scatter(new_pos[:,0],eval_result)

# +
# print(eval_result)
# -

"""
Result using finite difference method
"""

# +
import pandas as pd
FD_file = pd.read_excel('FD_beta20.xlsx')

points = pd.DataFrame(FD_file['X'])
target = pd.DataFrame(FD_file['Y'])
plt.plot(points,target,label = 'q_true')
plt.scatter(new_pos[:,0],eval_result,color = 'C1',label = 'q_TT',s = 10)
plt.legend()
plt.xlabel('x1')
plt.ylabel('committor function')
plt.show()
# -

target_np = np.array(target.values)
target_new = np.repeat(target_np,216)

err_vec = target_new-eval_result
print(err_vec.shape)

plt.plot(new_pos[:,0],err_vec)
plt.title('error: q_true - q')
plt.xlabel('x1')
plt.ylabel('Residual')

# relative error
norm2_err = np.linalg.norm(err_vec,2)
norm2_target = np.linalg.norm(target_np,2)
relative_error_4D = norm2_err/norm2_target
print(relative_error_4D)


# MSE
square_errer_4D = np.square(err_vec)
RMSE_4D = np.sqrt(square_errer_4D.mean())
print(RMSE_4D)

"""
Testing points in 10D and plot the estimation by the computed result
"""

D = 4
x = np.linspace(-1,1,D)
x2 = np.linspace(-1,1,41)
x3 = np.linspace(-1,1,D)
x4 = np.linspace(-1,1,D)
x5 = np.linspace(-1,1,D)
x6 = np.linspace(-1,1,D)
x7 = np.linspace(-1,1,D)
x8 = np.linspace(-1,1,D)
x9 = np.linspace(-1,1,D)
x10 = np.linspace(-1,1,D)
YY,XX,ZZ,X4,X5,X6,X7,X8,X9,X10 = np.meshgrid(*[x,x2,x3,x4,x5,x6,x7,x8,x9,x10])
# print('YY',YY)
# print('XX',XX)
# print('ZZ',ZZ)
positions2 = np.vstack((XX.ravel(),YY.ravel(),ZZ.ravel(),X4.ravel(),X5.ravel(),X6.ravel(),X7.ravel(),X8.ravel(),X9.ravel(),X10.ravel()))
print(positions2.shape)
new_pos = np.transpose(positions2)
print(new_pos.shape)

datapoints = pd.DataFrame(new_pos,columns=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10'])
datapoints.to_pickle('datapoints_10D')

beta = 5
eval_result = eval_Q(beta,new_pos,'Q_4D_beta5')
import matplotlib.pyplot as plt
plt.scatter(new_pos[:,0],eval_result)

# +
# print(eval_result)
# -

"""
Result using finite difference method
"""

# +
import pandas as pd
FD_file = pd.read_excel('FD.xlsx')

points = pd.DataFrame(FD_file['X'])
target = pd.DataFrame(FD_file['Y'])
plt.plot(points,target,label = 'q_true')
plt.scatter(new_pos[:,0],eval_result,color = 'C1',label = 'q_TT',s = 10)
plt.legend()
plt.xlabel('x1')
plt.ylabel('committor function')
plt.show()
# -

target_np = np.array(target.values)
target_new = np.repeat(target_np,216)

# +
# x = np.linspace(-1,1,41)
# y = np.linspace(-1,1,41)
# YY,XX = np.meshgrid(x,y)
# positions2 = np.vstack((XX.ravel(),YY.ravel()))
# new_pos = np.transpose(positions2)
# beta = 20
# eval_result = eval_Q(beta,new_pos)
# -

err_vec = target_new-eval_result
print(err_vec.shape)

plt.plot(new_pos[:,0],err_vec)
plt.title('error: q_true - q')
plt.xlabel('x1')
plt.ylabel('Residual')

# relative error
norm2_err = np.linalg.norm(err_vec,2)
norm2_target = np.linalg.norm(target_np,2)
relative_error_4D = norm2_err/norm2_target
print(relative_error_2D)

# MSE
square_errer_4D = np.square(err_vec)
RMSE_4D = np.sqrt(square_errer_4D.mean())
print(RMSE_2D)


