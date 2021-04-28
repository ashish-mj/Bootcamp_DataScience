import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('Data/USA_Housing.csv')
print(data.head(30))
print(data.info())
print(data.describe())
#sns.pairplot(data)
sns.distplot(data['Price'])
sns.heatmap(data.corr(),annot=True)
