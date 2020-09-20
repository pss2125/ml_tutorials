#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:59:37 2020

@author: siva
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

#train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#svm classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train,y_train)

#predict
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#visualizing train
from  matplotlib.colors import ListedColormap
X1, X2 = np.meshgrid(np.arange(start= X_train[:,0].min()-1, stop=X_train[:,0].max()+1, step = 0.01),
                     np.arange(start=X_train[:,1].min()-1, stop=X_train[:,1].max()+1, step=0.01))


plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=.75, cmap=ListedColormap(('red','yellow')))

for i,j in enumerate(np.unique(y_train)):
    plt.scatter(X_train[y_train == j,0], X_train[y_train == j,1],
                c = ListedColormap(('red','yellow'))(i), label=j)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.title("svm training set")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

#visualizing test result
X1, X2 = np.meshgrid(np.arange(start= X_test[:,0].min()-1, stop=X_test[:,0].max()+1, step = 0.01),
                     np.arange(start=X_test[:,1].min()-1, stop=X_test[:,1].max()+1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=.75, cmap=ListedColormap(('red','yellow')))

for i,j in enumerate(np.unique(y_test)):
    plt.scatter(X_test[y_test == j,0], X_test[y_test == j,1],
                c = ListedColormap(('blue','yellow'))(i), label=j)
    
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
plt.title("svm test set")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()