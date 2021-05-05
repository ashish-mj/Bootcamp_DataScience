#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:44:51 2021

@author: ashish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier



pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


data = pd.read_csv('Data/kyphosis.csv')
print(data.head())

sns.pairplot(data,hue='Kyphosis')

X = data.drop('Kyphosis',axis=1)
Y = data['Kyphosis']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,Y_train)
predictions = dtree.predict(X_test)

print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
print(dtree.score(X_test,Y_test))

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,Y_train)
rfc_pred = rfc.predict(X_test)

print(classification_report(Y_test,rfc_pred))
print(confusion_matrix(Y_test,rfc_pred))
print(rfc.score(X_test,Y_test))


