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

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')