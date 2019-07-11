# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:42:59 2019

@author: CEM
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#outlook parçalara ayrılıp 0,1 değerlerini aldı
from sklearn.preprocessing import LabelEncoder
veriler=pd.read_csv("C:/Users/CEM/Desktop/odev_tenis.csv")
veriler2=veriler.apply(LabelEncoder().fit_transform)
# c = outlook
c=veriler2.iloc[:,:1]

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
c = ohe.fit_transform(c).toarray()
# windy play outlook temp humity atandı
havadurumu = pd.DataFrame(data=c,index=range(14),columns=['o','r','s'])
sonveriler=pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonveriler=pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regresor=LinearRegression()

regresor.fit(x_train,y_train)
y_pred=regresor.predict(x_test)
print(y_pred)

#backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)
X_l=sonveriler.iloc[:,[0,1,2,3,4,5]].values #sonuncu bağımlı 6 kolon
r_ols=sm.OLS(endog=sonveriler.iloc[:,-1:],exog=X_l)
r=r_ols.fit()
print(r.summary())
#print(r.summary) önemli en yüksek p>t olanı çıkar
sonveriler=sonveriler.iloc[:,1:] #çıkarıldı

#backward elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)
X_l=sonveriler.iloc[:,[0,1,2,3,4]].values #sonuncu bağımlı 6 kolon
r_ols=sm.OLS(endog=sonveriler.iloc[:,-1:],exog=X_l)
r=r_ols.fit()
print(r.summary())
x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]

regresor.fit(x_train,y_train)
y_pred=regresor.predict(x_test)
