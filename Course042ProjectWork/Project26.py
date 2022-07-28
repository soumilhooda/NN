#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 19:45:50 2022

@author: soumilhooda
"""

from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

boston = datasets.load_boston()
X = boston.data
y = boston.target

ss = StandardScaler()
X = ss.fit_transform(X)

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2)

model = Sequential()
model.add(Dense(13,input_dim=13,activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])

history = model.fit(Xtrain, ytrain, epochs=150, batch_size=10)
ypred=model.predict(Xtest)
ypred=ypred[:,0]

error=np.sum(np.abs(ytest-ypred))/np.sum(np.abs(ytest))*100
print('Prediction Error is',error,'%')