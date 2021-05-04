#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:09:51 2021

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


train = pd.read_csv('Data/titanic_train.csv')
print(train.head())

test = pd.read_csv('Data/titanic_test.csv')
print(test.head())


#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#sns.countplot(x='Survived',hue='Pclass',data=train)
#sns.distplot(train['Age'].dropna(),kde=False,bins=30)
#sns.boxplot(x='Pclass',y='Age',data=train)


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1: 
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
    
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)

#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
print(train.info())

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first = True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.drop('PassengerId',axis=1,inplace=True)
print(train.head())


X = train.drop('Survived',axis=1)
Y = train['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
predictions = logmodel.predict(X_test)
print(predictions)
print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
print(logmodel.score(X_test,Y_test))






