# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:23:21 2019

@author: CEM
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

veriler = pd.read_excel("C:/Users/CEM/Desktop/Iris.xls")
x= veriler.iloc[:,0:4].values #bağımsız
y=veriler.iloc[:,4:].values #bağımlı

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
y_pred= logr.predict(X_test)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print("Logistic Reg.")
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,y_train)
y_pred= knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print("KN")
print(cm)

from sklearn.svm import SVC
svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred= svc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print("SVC")
print(cm)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Gaussian")
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print("DecisionTree")
print(cm)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)
y_pred= rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Random Forest")
print(cm)

