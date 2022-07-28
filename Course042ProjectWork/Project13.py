#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 20:35:48 2022

@author: soumilhooda
"""

from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

irisset = datasets.load_iris()

X = irisset.data
Y = irisset.target

pca = PCA(n_components =2)
Xp = pca.fit(X).transform(X)

plt.figure(1)
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y, cmap='jet',s=10)
plt.suptitle('Orignal Clusters')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()


GMM = GaussianMixture(n_components = 3)
GMM.fit(Xp)
Y_predG = GMM.predict(Xp)





plt.figure(2)
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y_predG, cmap='jet',s=10)
plt.suptitle('GMM Clusters')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()
