#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:33:09 2021

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


loans = pd.read_csv('Data/loan_data.csv')
print(data.head())


#loans[loans['credit.policy']==1]['fico'].hist(bins=35,color='blue',label='Credit Policy 1',alpha=0.6)
#loans[loans['credit.policy']==0]['fico'].hist(bins=35,color='red',label='Credit Policy 0',alpha=0.6)
#plt.legend()

#plt.figure(figsize=(11,7))
#sns.countplot(x='purpose',hue='not.fully.paid',data=loans)

#sns.jointplot(x='fico',y='int.rate',data=loans,color='green')

plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',col='not.fully.paid')



print(loans.info())

cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
print(final_data.head())

X = final_data.drop('not.fully.paid',axis=1)
Y = final_data['not.fully.paid']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,Y_train)
predictions = dtree.predict(X_test)

print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
print(dtree.score(X_test,Y_test))


rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,Y_train)
rfc_pred = rfc.predict(X_test)

print(classification_report(Y_test,rfc_pred))
print(confusion_matrix(Y_test,rfc_pred))
print(rfc.score(X_test,Y_test))

