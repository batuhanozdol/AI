# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:10:03 2019

@author: Batuhan
"""
#apriori breath first   eclat depth first
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
veriler = pd.read_csv("C:/Users/CEM/Desktop/Restaurant_Reviews.csv")

import re


import nltk      # dahil edilmeyecek kelimeler yüklenir 
from nltk.stem.porter import PorterStemmer
stopword = nltk.download('stopwords')
ps= PorterStemmer()
from nltk.corpus import stopwords

# veri önişleme
yorumlar = []
for i in range(1000):
    # harf olmayan yorum işaretleri kaldırılır ve her sözcük ayrıştırılır
    yorum = re.sub('[^a-zA-Z]',' ',veriler['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))]
    yorum = ' '.join(yorum)
    yorumlar.append(yorum)

# öznitelik çıkarımı bag of words    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000)
X=cv.fit_transform(yorumlar).toarray() #bağımsız
y=veriler.iloc[:,1].values #bağımlı liked or not

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
