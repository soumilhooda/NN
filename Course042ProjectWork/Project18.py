#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:24:09 2022

@author: soumilhooda
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

X,y = make_blobs(n_samples=2500, centers=4,n_features=2,random_state=10)
y_pred = KMeans(n_clusters=4,random_state=0).fit_predict(X)

cm = confusion_matrix(y, y_pred)

plt.figure()
plt.scatter(X[:,0],X[:,1],c=y_pred,cmap='jet',s=10)
plt.suptitle('K-Means Clusters')
plt.axis('tight')
plt.show()

