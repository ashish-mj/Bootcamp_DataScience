#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:09:22 2021

@author: ashish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


data = pd.read_csv('Data/advertising.csv')
print(data.head())

data['Age'].plot.hist(bins=30)
#sns.jointplot(x="Age",y="Area Income",data=data)
#sns.jointplot(x="Age",y="Daily Time Spent on Site",data=data,kind='kde')
#sns.pairplot(data,hue='Clicked on Ad')


X = data[['Daily Time Spent on Site',"Age","Area Income","Daily Internet Usage","Male"]]
Y = data['Clicked on Ad']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)


print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
print(logmodel.score(X_test,Y_test))




