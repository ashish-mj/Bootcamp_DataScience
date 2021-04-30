#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:45:31 2021

@author: ashish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


customer = pd.read_csv('Data/e_customers')
print(customer.head())
print(customer.describe())
print(customer.info())

sns.jointplot(data=customer,x='Time on Website',y='Yearly Amount Spent')
sns.jointplot(data=customer,x='Time on App',y='Yearly Amount Spent')
sns.jointplot(data=customer,x='Time on App',y='Length of Membership',kind='hex')
sns.pairplot(customer)
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customer)