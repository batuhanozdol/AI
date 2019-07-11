# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:24:23 2019

@author: CEM
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

veri = pd.read_csv("C:/Users/CEM/Desktop/maaslar.csv")

#dataframe slice
x=veri.iloc[:,1:2]
y=veri.iloc[:,-1]
#array dönüşümü
X=x.values
Y=y.values
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
regresor=LinearRegression()
regresor.fit(X,Y)

#2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#4. dereceden polinom
poly_reg3=PolynomialFeatures(degree=4)
x_poly3=poly_reg3.fit_transform(X)
lin_reg3=LinearRegression()
lin_reg3.fit(x_poly3,y)

#görselleştirme
plt.scatter(X,Y)
plt.plot(X,regresor.predict((X)))
plt.show()
plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()
plt.scatter(X,Y)
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)))
plt.show()

#tahminler
print(regresor.predict(11))
print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg3.predict(poly_reg3.fit_transform(11)))

#veri ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcek=sc1.fit_transform(X.reshape(-1,1))
sc2=StandardScaler()
y_olcek=sc2.fit_transform(Y.reshape(-1,1))

#support vector regression 
from sklearn.svm import SVR
svr_reg = SVR()
svr_reg.fit(x_olcek,y_olcek)
#görüntüleme
plt.scatter(x_olcek,y_olcek)
plt.plot(x_olcek,svr_reg.predict(x_olcek))
plt.show()
print(svr_reg.predict(11))

#karar ağacı ile tahmin
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor(random_state=0)
tree.fit(X,Y)
plt.scatter(X,Y)
plt.plot(X,tree.predict(X))
print(tree.predict(11))

#random forest
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(X,Y)
print(rf.predict(11))






