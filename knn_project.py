#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:55:33 2021

@author: ashish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


data = pd.read_csv('Data/KNN_Project_Data')
print(data.head())

#sns.pairplot(data,hue='TARGET CLASS')
scaler = StandardScaler()
scaler.fit(data.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(data.drop('TARGET CLASS',axis=1))
data_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
print(data_feat.head())


X = data_feat
Y = data['TARGET CLASS']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,Y_train)
predictions = knn.predict(X_test)

print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
print(knn.score(X_test,Y_test))

error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    predict = knn.predict(X_test)
    error_rate.append(np.mean(predict != Y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,linestyle='dashed',marker='o')

knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train,Y_train)
predictions = knn.predict(X_test)

print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
print(knn.score(X_test,Y_test))
