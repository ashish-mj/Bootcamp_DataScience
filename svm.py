import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.keys())

features = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

X = features
Y = cancer['target']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=101)

model = SVC()
model.fit(X_train,Y_train)

predictions = model.predict(X_test)

print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))







