#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 20:35:38 2022

@author: soumilhooda
"""

import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import matplotlib.pyplot as plt

M =10
N = 20
x = np.array([0,0,3.2,0,0,0,0,0,4.1,0,0,0,0,0,0,0,0,0,0,0])

#Q = nr.random(M,N);
Q = 2*nr.randint(0,2,(M,N))-1;
y = np.matmul(Q,x);

xOMP = np.zeros(N);
Qk = np.zeros((M,N));
r_curr = y;
eth = 2.5;
k = 0;
set_I = np.zeros(N);

while np.absolute(nl.norm(r_curr)**2) > eth:
    m_ind = np.argmax(np.absolute(np.matmul(np.transpose(Q),r_curr)));
    Qk[:,k]=Q[:,m_ind];
    x_ls = np.matmul(nl.pinv(Qk[:,0:k+1]),y);
    r_curr = y - np.matmul(Qk[:,0:k+1],x_ls);
    k = k+1
    
set_I_nz = set_I[0:k];
xOMP[set_I_nz.astype(int)]=x_ls;

plt.scatter(np.arange(1,N+1),x,marker="x");
plt.scatter(np.arange(1,N+1),xOMP,facecolors='none',edgecolors='g');
plt.grid(1, which='both')  
plt.suptitle('Performance of QMP')
plt.legend(["True","Estimated"],loc='upper right');
plt.xlabel('Index')
plt.ylabel('Value')   
