#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 18:04:51 2022

@author: soumilhooda
"""
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Flatten, Dense
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from keras.models import Sequential
nc = 10 #number of classes

(Xtrain, ytrain), (Xtest, ytest) = fashion_mnist.load_data()

plt.figure(1)
imgplot1 = plt.imshow(Xtrain[nr.randint(60000)])
plt.show()
                      
plt.figure(2)
imgplot2 = plt.imshow(Xtrain[nr.randint(60000)])
plt.show()

Xtrain = Xtrain.reshape(60000,28,28,1)
Xtest=Xtest.reshape(10000,28,28,1)

ytrainEnc =tf.one_hot(ytrain,depth=nc)
ytestEnc = tf.one_hot(ytest,depth=nc)

model = Sequential()
model.add(Conv2D(64,kernel_size=3,activation="relu",input_shape=(28,28,1)))
model.add(Conv2D(32,kernel_size=3,activation="relu"))
model.add(Flatten())
model.add(Dense(10,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(Xtrain,ytrainEnc,validation_data=(Xtest,ytestEnc),epochs=3)

ypred = model.predict(Xtest)
ypred = np.argmax(ypred,axis=1)