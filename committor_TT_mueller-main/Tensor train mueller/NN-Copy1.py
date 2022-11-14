"""
Store the function psi^{k},a^{k},b^{k} for each dimension k (only one function to be returned)

For the double well potential function
"""


import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

# +
import math
pitorch = torch.Tensor([math.pi])


def dU(x,y):
    dUy = 0.6*y
    dUx = 4*((x.pow(2) - 1).pow(2))*x
    return dUx, dUy


# +
#Euler-Maruyama method with x0 = (0,0)

x0 = torch.tensor([[0]])
y0 = torch.tensor([[0]])
delt = torch.tensor(1e-2) #step size
Temp = torch.tensor(1)
Nt = int(40/delt)

# torch.manual_seed(1)

w = torch.randn(2, Nt)
w = torch.sqrt(delt)*w

#print(w)

# +
Xi = torch.tensor([]) 
Yi = torch.tensor([])
# all the points obtained from Euler-Maruyama scheme

train_data = torch.tensor([])
# all the points in omega\(A cup B)

a = torch.tensor([-1,0]) 
b = torch.tensor([1,0])
# Centers of circles of regions A and B 

rsquare = torch.tensor([0.1]).pow(2)
# r squared of circles A and B 
# -

for i in range(Nt):
    if i == 0:
        dUx,dUy = dU(x0,y0)
        newX = x0 - dUx*delt + torch.sqrt(2*Temp)*w[0][i]
        newY = y0 - dUy*delt + torch.sqrt(2*Temp)*w[1][i]
    else:
        dUx,dUy = dU(newX,newY)
        newX = newX - dUx*delt + torch.sqrt(2*Temp)*w[0][i]
        newY = newY - dUy*delt + torch.sqrt(2*Temp)*w[1][i]
        
    if newY > 1:
        newY = newY - torch.tensor(2)
    if newY < -1:
        newY = newY + torch.tensor(2)
    
    if i%1 == 0:
        if newX > -1 and newX < 1:
            Xi = torch.cat((Xi,newX), 0)
            Yi = torch.cat((Yi,newY), 0)



# +
train_data = torch.cat((Xi, Yi), 1)

print(train_data.shape)

# +
import torch
import matplotlib.pyplot as plt

Xx = train_data[:,0]
Yy = train_data[:,1]

plt.scatter(Xx.detach().numpy(), Yy.detach().numpy())
plt.scatter(a[0],a[1])
plt.scatter(b[0],b[1])
# -

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class Ruggedmueller2(nn.Module):
    """Feedfoward neural network with 2 hidden layer"""
    def __init__(self, in_size, hidden_size,hidden_size2, out_size):
        super().__init__()
        # 1st hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # 2nd hidden layer
        self.linear2 = nn.Linear(hidden_size,hidden_size2)
        # output layer
        self.linear3 = nn.Linear(hidden_size2, out_size)
        
    def forward(self, xb):
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        tanhf = nn.Tanh()
        out = tanhf(out)
        # Get predictions using output layer
        out = self.linear2(out)
        # apply activation function again
        out = tanhf(out)
        # last hidden layer 
        out = self.linear3(out)
        #sigmoid function
        out = torch.sigmoid(out)
        return out


def chiAB(X):
    x = X[:,0]
    a = torch.tensor([-1])
    b = torch.tensor([1])
    m = nn.Tanh()
    sizex, nothing = X.shape
    chiA = 0.5 - 0.5*m(1000*((torch.abs(x - a)).reshape(sizex,1)))
    chiB = 0.5 - 0.5*m(1000*((torch.abs(x - b)).reshape(sizex,1))) 
#     chiB = 0.5 - 0.5*m(1000*((((x - b).pow(2)).reshape(sizex,1))-(torch.tensor(0.02)).pow(2)))       
                             
    return chiA, chiB


def q_theta(X,chiA,chiB,q_tilde):
    Q = (torch.tensor([1]) - chiA)*(q_tilde*(torch.tensor([1]) - chiB)+chiB)
    return Q


def funU(x,y):
    U = (x.pow(2) - 1).pow(2) + 0.3*y.pow(2)
    
    return U


# +
N_neuron = 20
input_size = 2
output_size = 1

model = Ruggedmueller2(input_size,N_neuron,N_neuron,output_size)
# -

train_data.requires_grad_(True)
from torch.utils.data import TensorDataset
size1,size2 = train_data.shape
rhs = torch.zeros(size1,)
train_ds = TensorDataset(train_data,rhs)

# +
from torch.utils.data import DataLoader

batch_size = int(size1/125)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

import torch.optim as optim

loss_fn = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=5e-5)
#optimizer = optim.Adam([w1,w2,b1,b2], lr=1e-5)
schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,1500,2000,2500,3000,3500], gamma=0.5)
# -

import time

# +
Temp_committor = torch.tensor(0.2)
Temp = torch.tensor(1)
beta = 1/Temp_committor
betap = 1/Temp
loss = 1

start = time.time()

for epoch in range(4000):
    if epoch%100 == 0:
        print(epoch)
        print('loss: ',loss)
    for X,y in train_dl:
        optimizer.zero_grad()
        
        chia,chib = chiAB(X)
        q_tilde = model(X)
        Q = q_theta(X,chia,chib,q_tilde)
        
        U = funU(X[:,0],X[:,1])
    
        derivQ = torch.autograd.grad(Q,X,allow_unused=True, retain_graph=True, grad_outputs = torch.ones_like(Q), create_graph=True)
        #output = ((derivQ[0][0][0]).pow(2)+(derivQ[0][:][1]).pow(2))*torch.exp(-(beta - betap)*U)
        output = (torch.norm(derivQ[0],dim=1)**2)*torch.exp(-(beta - betap)*U)
        
        loss = loss_fn(output,y)
        loss.backward()
        optimizer.step()
    schedule.step()
    
end = time.time()
        
# -

print(end - start)

# +
import pandas as pd
chia_t,chib_t = chiAB(train_data)

q_tilde = model(train_data)

Q = q_theta(train_data,chia_t,chib_t,q_tilde)

Q_np = Q.detach().numpy()

Qf = pd.DataFrame(Q_np)
        
# Qf.to_excel('double_potential_wells.xlsx',index=False) #save to file

# -

import matplotlib.pyplot as plt
plt.scatter(train_data[:,0].detach().numpy(),Q_np)

# +
import pandas as pd
FD_file = pd.read_excel('FD.xlsx')

points = pd.DataFrame(FD_file['X'])
target = pd.DataFrame(FD_file['Y'])
plt.plot(points,target,label = 'q_true')
plt.scatter(train_data[:,0].detach().numpy(),Q_np,color = 'C1',label = 'q_NN',s = 10)
plt.legend()
plt.xlabel('x1')
plt.ylabel('committor function')
plt.title('comparison with NN')
plt.show()
# -

x = np.linspace(-1,1,41)
y = np.linspace(-1,1,41)
YY,XX = np.meshgrid(x,y)
positions2 = np.vstack((XX.ravel(),YY.ravel()))
new_pos = np.transpose(positions2)
test_pts = torch.tensor(new_pos,dtype=torch.float32)
print(test_pts)

target_np = np.array(target.values)
target_new = np.repeat(target_np,41)
print(target_new.shape)

# +
chia_t,chib_t = chiAB(test_pts)

q_tilde = model(test_pts)

Q = q_theta(test_pts,chia_t,chib_t,q_tilde)

Q_np = np.reshape(Q.detach().numpy(),(1681,))
print(Q_np.shape)

err_vec = target_new-Q_np
print(err_vec.shape)
# -

plt.plot(points,target,label = 'q_true')
plt.scatter(new_pos[:,0],Q_np,color = 'C1',label = 'q_NN',s = 10)
plt.legend()
plt.xlabel('x1')
plt.ylabel('committor function')
plt.title('comparison with NN')
plt.show()

plt.plot(new_pos[:,0],err_vec)
plt.title('error: q_true - q_NN')
plt.xlabel('x1')
plt.ylabel('Residual')

# relative error
norm2_err = np.linalg.norm(err_vec,2)
norm2_target = np.linalg.norm(target_np,2)
relative_error_NN = norm2_err/norm2_target
print(relative_error_NN)

# MSE
square_errer_NN = np.square(err_vec)
RMSE_NN = np.sqrt(square_errer_NN.mean())
print(RMSE_NN)


