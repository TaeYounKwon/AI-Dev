import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=2022)

# Random Forest 
rf = RandomForestClassifier(max_depth=5)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Accuracy of Random Forest {:.2f}'.format(accuracy_score(y_test, y_pred)))