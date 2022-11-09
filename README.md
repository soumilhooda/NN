# NN
My early learning space for various ML/NN/RNN/CNN/DL/CV concepts.

Shall write descriptions after finishing my semester backlog. Mathematical Methods of Physics is getting the better of me. Update - It did end up getting the better of me :')

# Overview
Have implemented the following tasks from-scratch (i.e. without the use of libraries here);
- Linear Regression
  - LR with Batch Gradient Descent
  - LR with Mini Batch Gradient Descent
  - LR with Stochastic Gradient Descent
- Ridge Regression
  - RR with Batch Gradient Descent
  - RR with Mini Batch Gradient Descent
  - RR with Stochastic Gradient Descent
- Least Angle Regression
  - LAR with Batch Gradient Descent
  - LAR with Mini Batch Gradient Descent
  - LAR with Stochastic Gradient Descent
- Polynomial Regression
  - PR with Batch Gradient Descent
  - PR with Mini Batch Gradient Descent
  - PR with Stochastic Gradient Descent
- Logistic Regression
  - LOR with Batch Gradient Descent
  - LOR with Mini Batch Gradient Descent
  - LOR with Stochastic Gradient Descent
- Logistic Regression with L2 Norm Regularisation
  - LOR with Batch Gradient Descent and L2 Norm Regularisation
  - LOR with Mini Batch Gradient Descent and L2 Norm Regularisation
  - LOR with Stochastic Gradient Descent and L2 Norm Regularisation
- Logistic Regression with L1 Norm Regularisation
  - LOR with Batch Gradient Descent and L1 Norm Regularisation
  - LOR with Mini Batch Gradient Descent and L1 Norm Regularisation
  - LOR with Stochastic Gradient Descent and L1 Norm Regularisation
- Multiclass Classification using LOR
  - MCC using LOR with Hold Out Cross Validation
  - MCC using LOR with K Fold Cross Validation 
- Classifaction using MLE
- Classifcation using MAP
- Non-Linear Perceptron using Online (Hebbian) Learning Rule and Hold Out Cross Validation
- Kernel Perceptron with Holdout and K Fold Cross Validation
  - Linear Kernel
  - Radial Basis Function Kernel
  - Polynomial Kernel
- Radial Basis Function Neural Network with K Fold Cross Validation
- Stacked Auto Encoder based Neural Network with BPA and Hold Out Cross Validation
- Extreme Learning Machine with K Fold Cross Validation
  - Tan H Activation Function
  - Gaussian Activation Function
- Deep Layer Stacked Autoencoder based Extreme Learning Machine with Hold Out Cross Validation
- Support Vector Machine with SMO and Hold Out Cross validation
  - RBF Kernel
  - Polynomial Kernel

You can also find some good models that have been prepared using scikit-learn, TensorFlow/Keras and the usual Numpy, Matplotlib, Seaborn and Pandas. I have started using PyTorch very recently (circa March 2022). You may find work related to that in my other repository(s).

# Paper Implementation
As part of the 2nd Assignment for the course BITS F312 NNFL, we had to implement a paper by [Dr. Rajesh](https://www.bits-pilani.ac.in/Hyderabad/tripathyrk/Profile)'s student. The paper titled, [Detection of shockable ventricular cardiac arrhythmias from ECG signals using FFREWT filter-bank and deep convolutional neural network](https://www.sciencedirect.com/science/article/pii/S0010482520302742) utilised a network of the order given in the image below and it has been implemented using Tensorflow/Keras over [here](https://github.com/soumilhooda/MLDLNNtoCV/blob/main/Q9_NNFL_Assignment2_SoumilHooda.ipynb).

![The Model](/Images/Screenshot%202022-05-01%20at%209.36.52%20AM.png)

# Course 042 Project Description

- Project 10 : A simple Linear Regression fit on the IRIS dataset using scikit-learn.
- Project 11 : A simple Linear Regression fit on the Boston Housing Dataset using scikit-learn. Train R2 score = 0.73 and test R2 score = 0.73
- Project 12 : A simple Logistic Regression fit on some generic purchase data using scikit-learn.
- Project 13 : Principal Component Analysis on IRIS dataset using scikit-learn.
- Project 14 : Linear Discriminant Anaylsis on IRIS dataset using scikit-learn.
- Project 15 : Support Vector Machine based classification on IRIS dataset using scikit-learn.
- Project 16 : Linear SVM + Kernel SVM (RBF) classification on Breast Cancer Dataset using scikit-learn. Accuracy of Linear SVM = 97.2% and accuracy of Kernel SVM = 96.5%
- Project 17 : Naive Bayes implementation on some generic purchase data using scikit-learn.
- Project 18 : K Means Clustering implementaion on Isotropic Gaussian Blobs using scikit-learn.
- Project 19 : Gaussian Mixture Modeling based clustering implementation on Isotropic Gaussian Blobs using scikit-learn.
- Project 20 : Orthogonal Matching Pursuit implementation using only Numpy and Matplotlib.
- Project 21 : Decision Tree Classifier on IRIS dataset using scikit-learn.
- Project 22 : Decision Tree Classifier on generic purchase dataset using scikit-learn.
- Project 23 : Factor Analysis on IRIS dataset using scikit-learn.
- Project 24 : Factor Analysis on BFI datset using FactorAnalyzer.
- Project 25 : Independent Component Anaslysis using scikit-learn.
- Project 26 : Neural Network with single hidden layer implemented on Boston Housing Dataset using scikit-learn and Keras. Prediction error = 12.9%
- Project 27 : Neural Network with two hidden layers implemented on Mobile Prices dataset using scikit-learn and Keras. Accuracy score = 94.5%
- Project 28 : Convolutional Neural Network implemented on MNIST Fashion dataset using scikit-learn and Keras.
- Project 29 : Convolutional Neural Network implemented on MNIST Hand Written Digits dataset using scikit-learn and Keras.
- Project 30 : Neural Network with three hidden layers implemented on IMDB dataset using scikit-learn and Keras. Accuracy score = 85.1%
- Project 31 : Deep Learning Network implemented on CIFR dataset to classify 10 objects using scikit-learn and Keras.
- Project 32 : Case Study on Wine dataset. Implementation of the following classifiers and their respective accuracy scores.
  - Decision Tree Classifier = 93.3%
  - Random Forest Classifier = 97.7%
  - Naive Bayes Classifier = 93.3%
  - Nearest Neighbour Classifier = 97.7%
  - Linear SVM Classifier = 100%
  - Ridge Classifier = 100%
  - Logistic Regression = 100%
- Project 33 : Case Study on Breast Cancer dataset. Implementation of the following classifiers and their respective accuracy scores.
  - Random Forest Classifier = 97.2%
  - Decision Tree Classifier = 93.7%
  - Naive Bayes Classifier = 91.6%
  - Nearest Neighbour Classifier = 94.4%
  - Linear SVM Classifier = 97.2%
  - RBF Kernel SVM Classifier = 96.5%
  - Polynomial Kernel SVM Classifier = 91.6%
  - Sigmoid Kernel SVM Classifier = 95.1% 
  - Logistic Regression = 96.5%
