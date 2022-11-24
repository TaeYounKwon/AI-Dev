import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



iris = load_iris()

# 1. Check data info, type, structure, 
# print(iris.DESCR)

data = iris.data
label = iris.target
columns = iris.feature_names

data = pd.DataFrame(data, columns=columns)
data.head()

# Data Setup

X_train, X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=2022)

# Logistic Regression
lr = LogisticRegression()

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print('Accuracy of Logistic Regression: {:.2f}'.format(accuracy_score(y_test, y_pred)))
