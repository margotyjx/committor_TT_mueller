from scipy import integrate
import numpy as np
from numpy import exp
from scipy import special
import math
import Q_init,orth_pol_px,orth_pol_q,contraction,retraction
import evaluation
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# +
# comp = contraction.Component_easier(5,3,4)
# HA = comp.HA()
# print(HA[0].shape)

# +
# A = HA[-1].reshape(1,5,5,1)
# Qi = np.random.rand(2,5,2)
# print(Qi.shape)
# -



# +
class ALS_update:
    def __init__(self,beta,d,max_pol_deg,max_pol_deg_px,rank,rho,retrieve,filename = 'Q'):
        self.beta = beta
        self.max_pol_deg = max_pol_deg
        self.d = d
        self.max_pol_deg_vec = [max_pol_deg]*d
        self.max_pol_deg_vec_px = [max_pol_deg_px]*2 + [max_pol_deg]*(d-2)
        self.pol_deg = self.max_pol_deg + 1
        self.pol_deg_px = max_pol_deg_px+1
        self.rank = rank
        self.rho = rho 
        
        self.a1 = -1.5
        self.a2 = 1
        self.b1 = -0.5
        self.b2 = 2
        
        self.all_phi = orth_pol_q.all_pol_basis(self.max_pol_deg_vec,d,beta)

        self.pab = orth_pol_px.orth_pab(beta,d,max_pol_deg,max_pol_deg_px)
        self.comp = contraction.Component(beta,d,max_pol_deg,max_pol_deg_px)
        self.HA = self.comp.HA()
        self.HB = self.comp.HB()
        self.hb = self.comp.hb()
        self.Hk = []
        
        for k in range(self.d):
            self.Hk.append(self.comp.Hk(k))
        
      
        # initialization
        if retrieve == True:
            self.Q = pickle.load(open(filename, 'rb'))
        else:
            self.Q = Q_init.init_Q_np(self.rank,self.d,self.max_pol_deg)  # WE CAN ONLY UPDATE Q
        
        
#         self.Q = pickle.load(open('Q_2D_beta5', 'rb'))
        #stack for boundary conditions
        self.leftAStack = []
        self.rightAStack = []
        
        self.leftBStack = []
        self.rightBStack = []
        
        self.leftHkStack = []
        self.rightHkStack = []
        
        self.left_hb_Stack = []
        self.right_hb_Stack = []
        
        
    def push_left_stack_A(self,position):
        if position == 0:
            curr_HA = self.HA[0].reshape(self.pol_deg,self.pol_deg,1)
            Qi = self.Q[0]
            temp_L = np.einsum('ij,ikm->jmk',Qi,curr_HA)
            temp_L = np.einsum('jmk,kq->jmq',temp_L,Qi)
            self.leftAStack.append(temp_L)
            # note that the temp_L has dimension (rank, 1,rank)
        else:
            curr_HA = self.HA[position]
            Qi = self.Q[position]
            temp_L = np.einsum('ijk,ljpq->ilpkq',Qi,curr_HA)
            temp_L = np.einsum('xpz,ilpkq->ilxkqz',Qi,temp_L)
            left_stack = self.leftAStack[-1]
            temp_L = np.einsum('ilx,ilxkqz->kqz',left_stack,temp_L)
            self.leftAStack.append(temp_L)
            
    def push_left_stack_B(self,position):
        if position == 0:
            curr_HB = self.HB[0].reshape(self.pol_deg,self.pol_deg,1)
            Qi = self.Q[0]
            temp_L = np.einsum('ij,ikm->jmk',Qi,curr_HB)
            temp_L = np.einsum('jmk,kq->jmq',temp_L,Qi)
            self.leftBStack.append(temp_L)
            # note that the temp_L has dimension (rank, 1,rank)
        else:
            curr_HB = self.HB[position]
            Qi = self.Q[position]
            temp_L = np.einsum('ijk,ljpq->ilpkq',Qi,curr_HB)
            temp_L = np.einsum('xpz,ilpkq->ilxkqz',Qi,temp_L)
            left_stack = self.leftBStack[-1]
            temp_L = np.einsum('ilx,ilxkqz->kqz',left_stack,temp_L)
            self.leftBStack.append(temp_L)
            
            
    def push_left_stack_Hk(self,position):
        temp_stack_Hk = []
        if position == 0:
            for k in range(self.d):
                Hk = self.Hk[k]
                curr_Hk = Hk[0].reshape(self.pol_deg,self.pol_deg,self.pol_deg_px)
                Qi = self.Q[0]
                temp_L = np.einsum('ij,ikm->jmk',Qi,curr_Hk)
                temp_L = np.einsum('jmk,kq->jmq',temp_L,Qi)
                
                temp_stack_Hk.append(temp_L) # will be k of it
            
            self.leftHkStack.append(temp_stack_Hk)
            # note that the temp_stack_Hk has dimension (k,rank,1,rank)
        else:
            for k in range(self.d):
                Hk = self.Hk[k]
                curr_Hk = Hk[position]
                Qi = self.Q[position]
                temp_L = np.einsum('ijk,ljpq->ilpkq',Qi,curr_Hk)
                temp_L = np.einsum('xpz,ilpkq->ilxkqz',Qi,temp_L)
                left_stack = self.leftHkStack[-1][k]
                temp_L = np.einsum('ilx,ilxkqz->kqz',left_stack,temp_L)
                
                temp_stack_Hk.append(temp_L)
                
            self.leftHkStack.append(temp_stack_Hk)
    def push_left_stack_hb(self,position):
        if position == 0:
            curr_hb = self.hb[0].reshape(self.pol_deg,1)
            Qi = self.Q[0]
            temp_L = np.einsum('ij,ik->jk',curr_hb,Qi)
            
            self.left_hb_Stack.append(temp_L)
            
        else:
            curr_hb = self.hb[position]
            Qi = self.Q[position]
            temp_L = np.einsum('ijk,ljq->ilkq',curr_hb,Qi)
            left_stack = self.left_hb_Stack[-1]
            temp_L = np.einsum('il,ilkq->kq',left_stack,temp_L)
            
            self.left_hb_Stack.append(temp_L)
    
    def push_right_stack_A(self, position):
        if position == self.d-1:
            curr_HA = self.HA[self.d-1].reshape(1,self.pol_deg,self.pol_deg)
            Qi = self.Q[self.d-1]
            temp_R = np.einsum('ji,kjm->ikm',Qi,curr_HA)
            temp_R = np.einsum('ikm,mj->ikj',temp_R,Qi)
            self.rightAStack.append(temp_R)
            
        else:
            curr_HA = self.HA[position]
            Qi = self.Q[position]
            
            temp_R = np.einsum('ijk,ljpq->ilpkq',Qi,curr_HA)
            temp_R = np.einsum('xpz,ilpkq->ilxkqz',Qi,temp_R)
            right_stack = self.rightAStack[-1]
            temp_R = np.einsum('ilxkqz,kqz->ilx',temp_R,right_stack)
            self.rightAStack.append(temp_R)
            
            
    def push_right_stack_B(self, position):
        if position == self.d-1:
            curr_HB = self.HB[self.d-1].reshape(1,self.pol_deg,self.pol_deg)
            Qi = self.Q[self.d-1]
            temp_R = np.einsum('ji,kjm->ikm',Qi,curr_HB)
            temp_R = np.einsum('ikm,mj->ikj',temp_R,Qi)
            self.rightBStack.append(temp_R)
            
        else:
            curr_HB = self.HB[position]
            Qi = self.Q[position]
            
            temp_R = np.einsum('ijk,ljpq->ilpkq',Qi,curr_HB)
            temp_R = np.einsum('xpz,ilpkq->ilxkqz',Qi,temp_R)
            right_stack = self.rightBStack[-1]
            temp_R = np.einsum('ilxkqz,kqz->ilx',temp_R,right_stack)
            self.rightBStack.append(temp_R)
            
    def push_right_stack_hb(self, position):
        if position == self.d-1:
            curr_hb = self.hb[self.d-1].reshape(1,self.pol_deg)
            Qi = self.Q[self.d-1]
            temp_R = np.einsum('ji,ik->jk',curr_hb,Qi)
            
            self.right_hb_Stack.append(temp_R)   
        else:
            curr_hb = self.hb[position]
            Qi = self.Q[position]
            
            temp_R = np.einsum('ijk,ljq->ilkq',curr_hb,Qi)
            right_stack = self.right_hb_Stack[-1]
            temp_R = np.einsum('ilkq,kq->il',temp_R,right_stack)
            
            self.right_hb_Stack.append(temp_R)
            
    def push_right_stack_Hk(self, position):
        temp_stack_Hk = []
        if position == self.d-1:
            for k in range(self.d):
                Hk = self.Hk[k]
                if self.d == 2:
                    curr_Hk = Hk[self.d-1].reshape(self.pol_deg_px,self.pol_deg,self.pol_deg)
                else:
                    curr_Hk = Hk[self.d-1].reshape(1,self.pol_deg,self.pol_deg)
                    
                Qi = self.Q[self.d-1]
                temp_R = np.einsum('ji,kjm->ikm',Qi,curr_Hk)
                temp_R = np.einsum('ikm,mj->ikj',temp_R,Qi)
                temp_stack_Hk.append(temp_R)
            
            self.rightHkStack.append(temp_stack_Hk)
            
        else:
            for k in range(self.d):
                Hk = self.comp.Hk(k)
                curr_Hk = Hk[position]
                Qi = self.Q[position]
            
                temp_R = np.einsum('ijk,ljpq->ilpkq',Qi,curr_Hk)
                temp_R = np.einsum('xpz,ilpkq->ilxkqz',Qi,temp_R)
                right_stack = self.rightHkStack[-1][k]
                temp_R = np.einsum('ilxkqz,kqz->ilx',temp_R,right_stack)
                
                temp_stack_Hk.append(temp_R)
                
            self.rightHkStack.append(temp_stack_Hk)
    
    
    def calc_error(self):
        for pos in range(self.d-1,0,-1):
            temp_stack_Hk = []
            if pos == self.d-1:
                # Hk
                Qi = self.Q[self.d-1]
                for k in range(self.d):
                    Hk = self.Hk[k]
                    if self.d == 2:
                        curr_Hk = Hk[self.d-1].reshape(self.pol_deg_px,self.pol_deg,self.pol_deg)
                    else:
                        curr_Hk = Hk[self.d-1].reshape(1,self.pol_deg,self.pol_deg)
                    temp_R = np.einsum('ji,kjm->ikm',Qi,curr_Hk)
                    temp_R = np.einsum('ikm,mj->ikj',temp_R,Qi)
                    temp_stack_Hk.append(temp_R)
                    
                right_stack_Hk = temp_stack_Hk
                
                # HA
                curr_HA = self.HA[self.d-1].reshape(1,self.pol_deg,self.pol_deg)
                temp_R = np.einsum('ji,kjm->ikm',Qi,curr_HA)
                temp_R = np.einsum('ikm,mj->ikj',temp_R,Qi)
                right_stack_A = temp_R
                
                # HB
                curr_HB = self.HB[self.d-1].reshape(1,self.pol_deg,self.pol_deg)
                temp_R = np.einsum('ji,kjm->ikm',Qi,curr_HB)
                temp_R = np.einsum('ikm,mj->ikj',temp_R,Qi)
                right_stack_B = temp_R
                
                #hb
                curr_hb = self.hb[self.d-1].reshape(1,self.pol_deg)
                temp_R = np.einsum('ji,ik->jk',curr_hb,Qi)

                right_stack_hb = temp_R
            else:
                Qi = self.Q[pos]
                # Hk
                for k in range(self.d):
                    Hk = self.comp.Hk(k)
                    curr_Hk = Hk[pos]

                    temp_R = np.einsum('ijk,ljpq->ilpkq',Qi,curr_Hk)
                    temp_R = np.einsum('xpz,ilpkq->ilxkqz',Qi,temp_R)
                    temp_R = np.einsum('ilxkqz,kqz->ilx',temp_R,right_stack_Hk[k])
                    temp_stack_Hk.append(temp_R)
                    
                right_stack_Hk = temp_stack_Hk
                
                #HA
                curr_HA = self.HA[pos]

                temp_R = np.einsum('ijk,ljpq->ilpkq',Qi,curr_HA)
                temp_R = np.einsum('xpz,ilpkq->ilxkqz',Qi,temp_R)
                temp_R = np.einsum('ilxkqz,kqz->ilx',temp_R,right_stack_A)
                
                right_stack_A = temp_R
                
                #HB
                curr_HB = self.HB[pos]

                temp_R = np.einsum('ijk,ljpq->ilpkq',Qi,curr_HB)
                temp_R = np.einsum('xpz,ilpkq->ilxkqz',Qi,temp_R)
                temp_R = np.einsum('ilxkqz,kqz->ilx',temp_R,right_stack_B)
                
                right_stack_B = temp_R
                
                #hb
                curr_hb = self.hb[pos]

                temp_R = np.einsum('ijk,ljq->ilkq',curr_hb,Qi)
                temp_R = np.einsum('ilkq,kq->il',temp_R,right_stack_hb)
                
                right_stack_hb = temp_R

            
                    
        # left most Hk
        Qi = self.Q[0]
        temp_left_stack_Hk = []
        for k in range(self.d):
            Hk = self.Hk[k]
            curr_Hk = Hk[0].reshape(self.pol_deg,self.pol_deg,self.pol_deg_px)
            temp_L = np.einsum('ij,ikm->jmk',Qi,curr_Hk)
            temp_L = np.einsum('kq,jmk->jmq',Qi,temp_L)

            temp_left_stack_Hk.append(temp_L) # will be k of it
            
        # left most HA
        curr_HA = self.HA[0].reshape(self.pol_deg,self.pol_deg,1)
        temp_L = np.einsum('ij,ikm->jmk',Qi,curr_HA)
        temp_L = np.einsum('kq,jmk->jmq',Qi,temp_L)
        
        left_stack_A = temp_L
        
        # left most HB
        curr_HB = self.HB[0].reshape(self.pol_deg,self.pol_deg,1)
        temp_L = np.einsum('ij,ikm->jmk',Qi,curr_HB)
        temp_L = np.einsum('kq,jmk->jmq',Qi,temp_L)
        
        left_stack_B = temp_L
        
        # left most hb
        curr_hb = self.hb[0].reshape(self.pol_deg,1)
        temp_L = np.einsum('ij,ik->jk',curr_hb,Qi)

        left_stack_hb = temp_L
        
        # sums up for Hk
        for k in range(self.d):
            left_k = temp_left_stack_Hk[k]
            right_k = right_stack_Hk[k]
            out_k = np.einsum('ijk,ijk->',left_k,right_k)
            if k==0:
                out_Hk = out_k
            else:
                out_Hk += out_k # should be a number
        # sums up for HA
        out_HA = np.einsum('ijk,ijk->',left_stack_A,right_stack_A)
        # sums up for HB
        out_HB = np.einsum('ijk,ijk->',left_stack_B,right_stack_B)
        # sums up for hb
        out_hb = np.einsum('ij,ij->',left_stack_hb,right_stack_hb)
        
        print('err HA ',out_HA)
        print('err HB ',out_HB - 2*out_hb + 1)
        print('err Hk ',out_Hk)
        
#         print('curr_HA',curr_HA)
#         print('curr_HB',curr_HB)
#         print('curr_hb',curr_hb)
        
        
        err = np.abs(out_Hk+self.rho*out_HA+self.rho*out_HB-self.rho*out_hb)
        
        
        return err
        
        
    
    
    def solve(self,max_iter):
        # build right stack for updating
        for pos in range(self.d-1,0,-1):
            self.push_right_stack_A(pos)
            self.push_right_stack_Hk(pos)
            self.push_right_stack_hb(pos)
            self.push_right_stack_B(pos)
        
        
        ## for iteration, check if conditions are met
        
        for itr in range(max_iter):
            err = self.calc_error()
#             if err < 0.005:
#                 print('error less than 0.005, and error is ',err)
# #                 break
#             else:
#                 print('error is: ',err)
                
            # begin the minimization
        
        
            for pos in range(0,self.d):
                print('pos: ',pos)
                if pos == 0:
                    I_curr = retraction.identity_tensor_first(self.pol_deg,self.rank)

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # HA:
                    R_curr_A = self.rightAStack[-1]
                    curr_HA = self.HA[0].reshape(self.pol_deg,self.pol_deg,1)
                    
                    op_A = np.einsum('jim,jpk,pqn->mikqn',I_curr,curr_HA,I_curr)
                    op_A = np.einsum('mikqn,ikq->mn',op_A,R_curr_A)


                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    #rhs:
                    R_curr_hb = self.right_hb_Stack[-1]
                    curr_hb = self.hb[0].reshape(self.pol_deg,1)
                    
                    rhs = np.einsum('ij,ikm->jkm',curr_hb,I_curr)
                    rhs = np.einsum('jkm,jk->m',rhs,R_curr_hb)
                    rhs = self.rho*rhs

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # HB:
                    R_curr_B = self.rightBStack[-1]
                    curr_HB = self.HB[0].reshape(self.pol_deg,self.pol_deg,1)
                    
                    op_B = np.einsum('jim,jpk,pqn->mikqn',I_curr,curr_HB,I_curr)
                    op_B = np.einsum('mikqn,ikq->mn',op_B,R_curr_B)


                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Hk

                    for k in range(self.d):
                        Hk = self.Hk[k]
                        R_curr_Hk = self.rightHkStack[-1][k]
                        curr_Hk = Hk[0].reshape(self.pol_deg,self.pol_deg,self.pol_deg_px)
                        
                        op_k = np.einsum('jim,jpk,pqn->mikqn',I_curr,curr_Hk,I_curr)
                        op_k = np.einsum('mikqn,ikq->mn',op_k,R_curr_Hk)


                        if k == 0:
                            op_Hk = op_k
                        else:
                            op_Hk += op_k


                    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    # solve the linear equation and update the components
                    size1,size2 = op_A.shape
                    op = op_Hk+ self.rho*op_A + self.rho*op_B
                    
                    sol = np.linalg.solve(op, rhs)
                    print('cond of op_Hk', np.linalg.cond(op_Hk))
                    print('pos 0 error: ', np.linalg.norm(op@sol - rhs))
                    print('relative error: ', np.linalg.norm(op@sol - rhs)/np.linalg.norm(rhs)*np.linalg.cond(op))

                    Qi = np.reshape(sol,(self.pol_deg,self.rank))
                    self.Q[pos] = Qi


                    self.push_left_stack_A(pos)
                    self.push_left_stack_hb(pos)
                    self.push_left_stack_B(pos)
                    self.push_left_stack_Hk(pos)

                    self.rightAStack.pop()
                    self.right_hb_Stack.pop()
                    self.rightBStack.pop()
                    self.rightHkStack.pop()


                elif pos == self.d-1:
                    # last node, right stack empty
                    I_curr = retraction.identity_tensor_first(self.pol_deg,self.rank)

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # HA
                    curr_HA = self.HA[pos].reshape(1,self.pol_deg,self.pol_deg)
                    L_curr_A = self.leftAStack[-1]
                    op_A = np.einsum('jim,kjp,pqn->mikqn',I_curr,curr_HA,I_curr)
                    op_A = np.einsum('ikq,mikqn->mn',L_curr_A,op_A)

                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # hb
                    curr_hb = self.hb[pos].reshape(1,self.pol_deg)
                    L_curr_hb = self.left_hb_Stack[-1]

                    rhs = np.einsum('ij,jkm->ikm',curr_hb,I_curr)
                    rhs = np.einsum('ik,ikm->m',L_curr_hb,rhs)
                    rhs = self.rho*rhs

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # HB
                    curr_HB = self.HB[pos].reshape(1,self.pol_deg,self.pol_deg)
                    L_curr_B = self.leftBStack[-1]
                    op_B = np.einsum('jim,kjp,pqn->mikqn',I_curr,curr_HB,I_curr)
                    op_B = np.einsum('ikq,mikqn->mn',L_curr_B,op_B)

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Hk
                    for k in range(self.d):
                        Hk = self.Hk[k]
                        L_curr_Hk = self.leftHkStack[-1][k]
                        if self.d == 2:
                            curr_Hk = Hk[pos].reshape(self.pol_deg_px,self.pol_deg,self.pol_deg)
                        else: 
                            curr_Hk = Hk[pos].reshape(1,self.pol_deg,self.pol_deg)

                        op_k = np.einsum('jim,kjp,pqn->mikqn',I_curr,curr_Hk,I_curr)
                        op_k = np.einsum('ikq,mikqn->mn',L_curr_Hk,op_k)

                        if k == 0:
                            op_Hk = op_k
                        else:
                            op_Hk += op_k



                    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # solve the linear equation and update components
                    size1,size2 = op_A.shape
                    op = op_Hk+self.rho*op_A + self.rho*op_B
                    
                    sol = np.linalg.solve(op, rhs)
                    print('cond of op_Hk: ', np.linalg.cond(op_Hk))
                    
                    print('pos 1 error: ', np.linalg.norm(op@sol - rhs))
                    print('relative error: ', np.linalg.norm(op@sol - rhs)/np.linalg.norm(rhs)*np.linalg.cond(op))

                    Qi = np.reshape(sol,(self.pol_deg,self.rank))
                    self.Q[pos] = Qi

                else:
                    I_curr = retraction.identity_tensor(self.rank,self.pol_deg,self.rank)
                    
                    # %%%%%%%%%%%%%%%%%%%%%%%%%%
                    # HA
                    R_curr_A = self.rightAStack[-1]
                    L_curr_A = self.leftAStack[-1]

                    curr_HA = self.HA[pos]
                    
                    op_A = np.einsum('ixpm,jxyq,kyzn->ijkpqzmn',I_curr,curr_HA,I_curr)
                    op_A = np.einsum('ijk,ijkpqzmn,pqz->mn',L_curr_A,op_A,R_curr_A)

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # rhs
                    R_curr_hb = self.right_hb_Stack[-1]
                    L_curr_hb = self.left_hb_Stack[-1]
                    curr_hb = self.hb[pos]

                    rhs = np.einsum('ijk,pjqm->ipkqm',curr_hb,I_curr)
                    rhs = np.einsum('ip,ipkqm,kq->m',L_curr_hb,rhs,R_curr_hb)
                    rhs = self.rho*rhs

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # HB
                    R_curr_B = self.rightBStack[-1]
                    L_curr_B = self.leftBStack[-1]

                    curr_HB = self.HB[pos]
                    
                    op_B = np.einsum('ixpm,jxyq,kyzn->ijkpqzmn',I_curr,curr_HB,I_curr)
                    op_B = np.einsum('ijk,ijkpqzmn,pqz->mn',L_curr_B,op_B,R_curr_B)

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # Hk
                    for k in range(self.d):
                        Hk = self.Hk[k]
                        L_curr_Hk = self.leftHkStack[-1][k]
                        R_curr_Hk = self.rightHkStack[-1][k]

                        curr_Hk = Hk[pos]
                        
                        op_k = np.einsum('ixpm,jxyq,kyzn->ijkpqzmn',I_curr,curr_Hk,I_curr)
                        op_k = np.einsum('ijk,ijkpqzmn,pqz->mn',L_curr_Hk,op_k,R_curr_Hk)


                        if k == 0:
                            op_Hk = op_k
                        else:
                            op_Hk += op_k

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # solve the linear equation
                    size1,size2 = op_A.shape
                    op = op_Hk + self.rho*op_A + self.rho*op_B
                    sol = np.linalg.solve(op, rhs)

                    Qi = np.reshape(sol,(self.rank,self.pol_deg,self.rank))

                    self.Q[pos] = Qi

                    self.push_left_stack_A(pos)
                    self.push_left_stack_hb(pos)
                    self.push_left_stack_B(pos)
                    self.push_left_stack_Hk(pos)

                    self.rightAStack.pop()
                    self.right_hb_Stack.pop()
                    self.rightBStack.pop()
                    self.rightHkStack.pop()


            # clear the left stack and update the right stack
            for pos in range(self.d-1,0,-1):
                self.push_right_stack_A(pos)
                self.push_right_stack_Hk(pos)
                self.push_right_stack_hb(pos)
                self.push_right_stack_B(pos)

                self.leftAStack.pop()
                self.left_hb_Stack.pop()
                self.leftBStack.pop()
                self.leftHkStack.pop()
        
            pickle.dump(self.Q, open("Q", 'wb'))
            
            
            FD_file = pd.read_excel('test_data.xlsx')

            points_x = pd.DataFrame(FD_file['X'])
            points_y = pd.DataFrame(FD_file['Y'])
            target = pd.DataFrame(FD_file['committor'])
            target_np = np.array(target.values)
            
            # evaluate the result
            
            new_pos = np.hstack((points_x.values,points_y.values))
            new_pos_trans = np.copy(new_pos)
            new_pos_trans[:,0] =(2*new_pos_trans[:,0] - (self.a1 + self.a2))/(self.a2 - self.a1)
            new_pos_trans[:,1] =(2*new_pos_trans[:,1] - (self.b1 + self.b2))/(self.b2 - self.b1)
            eval_result = evaluation.eval_Q(self.beta,new_pos_trans,'Q')
            err_vec = target_np-eval_result[:,None]
            
            square_errer_2D = np.square(err_vec)
            RMSE_2D = np.sqrt(square_errer_2D.mean())
            
            plt.scatter(new_pos[:,0],new_pos[:,1],c= eval_result,s = 20)
            plt.colorbar()
            plt.show()
            
            print(RMSE_2D)
            
#             self.rmse.append(RMSE_2D)
            
#             error_to_file = pd.DataFrame(np.array(self.rmse),columns = ['error'])
#             error_to_file.to_excel('error.xlsx',index = False)          
        
# -


## beta,d,max_pol_deg,max_pol_deg_px,rank,rho,retrieve
import time
start = time.time()
Als = ALS_update(1/100,2,25,25,1,50,False)
Als.solve(20)

# +
# end = time.time()
# print(end - start)
# -










