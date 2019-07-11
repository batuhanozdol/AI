# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:54:18 2019

@author: Batuhan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

veriler = pd.read_csv("C:/Users/CEM/Desktop/Wine.csv")
X=veriler.iloc[:,0:13].values
Y=veriler.iloc[:,13].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test= sc.fit_transform(x_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_train2 = pca.fit_transform(x_train)
x_test2 = pca.transform(x_test)

#pca dönüşüm öncesi
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

#pca dönüşümü sonrası
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(x_train2,y_train)

y_pred=classifier.predict(x_test)
y_pred2=classifier2.predict(x_test2)

from sklearn.metrics import confusion_matrix
# PCA olmadan
cm = confusion_matrix(y_test,y_pred)
print(cm)
# PCA sonrası
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)
# PCA sonrası PCA öncesi
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
x_train_lda = lda.fit_transform(x_train,y_train)
x_test_lda= lda.transform(x_test)

classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(x_train_lda,y_train)
y_pred_lda=classifier_lda.predict(x_test_lda)
#LDA ve orjinal
cm4 = confusion_matrix(y_pred,y_pred_lda)
print(cm4)





