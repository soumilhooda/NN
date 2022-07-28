#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:20:04 2022

@author: soumilhooda
"""

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression

dwine = datasets.load_wine()

X = dwine.data
Y = dwine.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

Xtrain, Xtest, Ytrain, Ytest \
    = train_test_split(X, Y, test_size =0.25, random_state=0)
    
#Decision Tree Classifier

dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(Xtrain, Ytrain)
Ypred = dtc.predict(Xtest)
dtcscore = accuracy_score(Ypred, Ytest)
print('Accuracy score of Decission Tree Classifier is',100*dtcscore,'%\n')

#Random Forest Classifier

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(Xtrain, Ytrain)
Ypred = rfc.predict(Xtest)
rfcscore = accuracy_score(Ypred, Ytest)
print('Accuracy score of Random Forest Classifier is',100*rfcscore,'%\n')

#Naive Bayes Classifier

nbc = GaussianNB()
nbc.fit(Xtrain, Ytrain)
Ypred = nbc.predict(Xtest)
nbcscore = accuracy_score(Ypred, Ytest)
print('Accuracy score of Naive Bayes Classifier is',100*nbcscore,'%\n')

#Nearest Neighbour Classifier

nnc = KNeighborsClassifier(n_neighbors=5)
nnc.fit(Xtrain, Ytrain)
Ypred = nnc.predict(Xtest)
nncscore = accuracy_score(Ypred, Ytest)
print('Accuracy score of Nearest Neighbor Classifier is',100*nncscore,'%\n')

#Multi Class SVM

svmc = LinearSVC(multi_class='crammer_singer',random_state=0)
svmc.fit(Xtrain, Ytrain)
Ypred = svmc.predict(Xtest)
svmcscore = accuracy_score(Ypred, Ytest)
print('Accuracy score of Linear SVM Classifier is',100*svmcscore,'%\n')

#Ridge Classifier

rcl = RidgeClassifier(random_state=0)
rcl.fit(Xtrain, Ytrain)
Ypred = rcl.predict(Xtest)
rclscore = accuracy_score(Ypred, Ytest)
print('Accuracy score of Ridge Classifier is',100*rclscore,'%\n')

#Logistic Regression

logr = LogisticRegression(multi_class='multinomial',random_state=0)
logr.fit(Xtrain, Ytrain)
Ypred = logr.predict(Xtest)
logrscore = accuracy_score(Ypred, Ytest)
print('Accuracy score of Logistic Regression is',100*logrscore,'%\n')

