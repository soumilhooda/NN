#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:00:52 2022

@author: soumilhooda
"""

from keras.datasets import imdb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

(Xtrain, ytrain), (Xtest, ytest) = imdb.load_data(num_words = 10000)

def vectorise(sequences, dimension = 10000):
    results = np.zeros((len(sequences),dimension))
    for i in range(len(sequences)):
        results[i,sequences[i]]=1.0
    return results


Xtrain = vectorise(Xtrain)
ytrain = np.array(ytrain).astype("float32")

model = Sequential()
model.add(Dense(50,input_dim=10000, activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

history = model.fit(Xtrain,ytrain,epochs=10,batch_size=550)

Xtest=vectorise(Xtest)
ypred = model.predict(Xtest)
ypred = np.round(ypred)

score=accuracy_score(ypred,ytest)
print('Accuracy score is',100*score,'%')
