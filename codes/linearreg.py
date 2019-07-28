# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

veri = pd.read_csv("C:/Users/CEM/Desktop/veriler.csv")
boy = veri[["boy"]]
boykilo=veri[["boy","kilo"]]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
Yas=veri.iloc[:,1:4].values
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])

ulke = veri.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
ulke = ohe.fit_transform(ulke).toarray()

c = veri.iloc[:,-1:].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")
c = ohe.fit_transform(c).toarray()

sonuc = pd.DataFrame(data=ulke,index = range(22) , columns=['fr','tr','us'])
sonuc2 = pd.DataFrame(data=Yas,index = range(22) , columns=['boy','kilo','yas'])
cinsiyet = veri.iloc[:,-1].values
sonuc3 = pd.DataFrame(data=c[:,:1],index=range(22),columns=['cinsiyet'])
s=pd.concat([sonuc,sonuc2],axis=1)
s2=pd.concat([s,sonuc3],axis=1)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
regresor=LinearRegression()
regresor.fit(x_train,y_train)
y_pred=regresor.predict(x_test)

boy=s2.iloc[:,3:4].values
sol=s2.iloc[:,:3]
sag=s2.iloc[:,4:]
veri=pd.concat([sol,sag],axis=1)
x_train,x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

r2=LinearRegression()
r2.fit(x_train,y_train)
y_pred=r2.predict(x_test)

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)
X_l=veri.iloc[:,[0,1,2,3,4,5]].values
r_ols=sm.OLS(endog=boy,exog=X_l)
r=r_ols.fit()

X_l=veri.iloc[:,[0,1,2,3,5]].values
r_ols=sm.OLS(endog=boy,exog=X_l)
r=r_ols.fit()
