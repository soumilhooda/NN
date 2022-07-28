#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:02:54 2022

@author: soumilhooda
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import FastICA

S = np.random.standard_t(1.5,size=(20000,2))

A = np.array([[1,1],[-1,2]])
X = np.dot(S, A.T)

ica = FastICA()
Sica = ica.fit(X).transform(X)

plt.figure(1)
plt.scatter(S[:,0],S[:,1],s=10,c='g')
plt.title('True Independent Sources')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

plt.figure(2)
plt.scatter(X[:,0],X[:,1],s=10,c='b')
plt.title('Mixed Source')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

plt.figure(3)
plt.scatter(Sica[:,0],Sica[:,1],s=10,c='r')
plt.title('ICA Recovered Signals')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()
