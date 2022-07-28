#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 20:18:18 2022

@author: soumilhooda
"""
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

irisset = datasets.load_iris()

X = irisset.data
Y = irisset.target

#LDA used for classification
lda1 = LinearDiscriminantAnalysis()
lda1.fit(X,Y)
Ypred1 = lda1.predict(X)
cmat1 = confusion_matrix(Y, Ypred1)


#LDA used for dimensionality reduction
lda2=LinearDiscriminantAnalysis(n_components=2)
Xl = lda2.fit(X,Y).transform(X)

plt.figure(1);
plt.scatter(Xl[:, 0], Xl[:, 1], c = Y)
plt.suptitle('LDA IRIS Data')
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()


#LDA used for classification after dimensionality reduction by LDA
lda3 = LinearDiscriminantAnalysis()
lda3.fit(Xl,Y)
Ypred3 = lda3.predict(Xl)
cmat3 = confusion_matrix(Y, Ypred3)


pca = PCA(n_components=2);
Xp = pca.fit(X).transform(X)

plt.figure(2);
plt.scatter(Xp[:, 0], Xp[:, 1], c = Y)
plt.suptitle('PCA IRIS Data')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

# LDA used for classification after dimensionality reduction by PCA
lda4 = LinearDiscriminantAnalysis()
lda4.fit(Xp, Y)
Ypred4 = lda4.predict(Xp)
cmat4 = confusion_matrix(Y, Ypred4)