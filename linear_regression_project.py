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

print(customer.columns)
X = customer[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
Y = customer['Yearly Amount Spent']

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(X_train, Y_train)
print(lm.coef_)

cdf = pd.DataFrame(lm.coef_,X.columns,columns=["Coeff"])
print(cdf)

predictions = lm.predict(X_test)
plt.scatter(Y_test,predictions)
plt.xlabel('Actual values')
plt.ylabel('Predicted Values')


print(metrics.mean_absolute_error(Y_test, predictions))   #mae
print(metrics.mean_squared_error(Y_test, predictions))    #mse
print(np.sqrt(metrics.mean_squared_error(Y_test, predictions)))  #rmse


print(lm.predict([[34.30555662975554,13.717513665142508,36.72128267790313,3.1201787827480914]]))

