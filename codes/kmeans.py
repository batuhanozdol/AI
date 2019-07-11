# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:02:46 2019

@author: Batuhan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

veriler = pd.read_csv("C:/Users/CEM/Desktop/musteri.csv")
X = veriler.iloc[:,3:].values    # kişi maaş ve hacim değerleri

#KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3,init='k-means++')
kmeans.fit(X)

sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
plt.plot(range(1,11),sonuclar)  # kümeleme rakamı için sonuç grafiği incelenir
plt.show()

kmeans = KMeans(n_clusters = 4,init='k-means++')
kmeans.fit_predict(X)
Y = kmeans.fit_predict(X)
plt.scatter(X[Y==0,0],X[Y==0,1],s=100,c='red')
plt.scatter(X[Y==1,0],X[Y==1,1],s=100,c='blue')
plt.scatter(X[Y==2,0],X[Y==2,1],s=100,c='green')
plt.scatter(X[Y==3,0],X[Y==3,1],s=100,c='yellow')
plt.show()

#Hierarchical
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4,affinity='euclidean', linkage='ward')
Y = ac.fit_predict(X)
plt.scatter(X[Y==0,0],X[Y==0,1],s=100,c='red')
plt.scatter(X[Y==1,0],X[Y==1,1],s=100,c='blue')
plt.scatter(X[Y==2,0],X[Y==2,1],s=100,c='green')
plt.scatter(X[Y==3,0],X[Y==3,1],s=100,c='yellow')
plt.show()

import scipy.cluster.hierarchy as hi
dendogram = hi.dendrogram(hi.linkage(X,method='ward'))
print('DENDOGRAM')
plt.show()

