# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:35:02 2019

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

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)

#yapay sinir ağı
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dense(6,init='uniform',activation='relu'))
classifier.add(Dense(1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,epochs=50)


y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.50) #true 1 false 0

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)








