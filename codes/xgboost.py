# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 23:19:41 2019

@author: Batuhan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
veriler = pd.read_csv("C:/Users/CEM/Desktop/Churn_Modelling.csv")

X=veriler.iloc[:,3:13].values
Y=veriler.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
le2=LabelEncoder()
X[:,2]=le2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)