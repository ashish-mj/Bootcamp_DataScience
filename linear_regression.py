import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Read Data
data = pd.read_csv('Data/USA_Housing.csv')
print(data.head(30))
print(data.info())
print(data.describe())
#sns.pairplot(data)
sns.distplot(data['Price'])
sns.heatmap(data.corr(),annot=True)


print(data.columns)
X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
Y = data['Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train, Y_train)
print(lm.intercept_)
print(lm.coef_)



