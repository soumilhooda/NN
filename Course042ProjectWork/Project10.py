#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:39:54 2022

@author: soumilhooda
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets

irisset = datasets.load_iris()
X = irisset.data[:50,:2]

reg = LinearRegression().fit(X[:,0:1],X[:,1])
w = reg.coef_
c = reg.intercept_

xpoints = np.linspace(4,7)
ypoints = w[0]*xpoints + c    
                                   



plt.plot(xpoints, ypoints, 'g-')
plt.scatter(X[:, 0], X[:, 1],s=10)
plt.suptitle('Linear Regression IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()