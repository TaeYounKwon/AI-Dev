import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


boston = load_boston()

# 1. Check data info, type, structure, 
# print(boston_dataset)

#print('Key of boston_dataset:\n',boston.keys())
#  Key of boston_dataset:
#  dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename', 'data_module'])

data = boston.data # or boston['data']
label = boston.target
columns = boston.feature_names




# 2. Data visualization
data = pd.DataFrame(data, columns=columns)

# print(data.head())
#  CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATIO       B  LSTAT
# 0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0     15.3  396.90   4.98
# 1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0     17.8  396.90   9.14
# 2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0     17.8  392.83   4.03
# 3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0     18.7  394.63   2.94
# 4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0     18.7  396.90   5.33

# print(data.shape)
# (506, 13)

# 3. 2nd Machine Learning Model: Simple Linear Regression
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2,random_state=2022) 
# We are using X_train['RM']
# But X_train['RM] is not enough to teach machine, need to be 2D+, But ['RM']is 1D
# add , blank by reshape(-1,1), -1,1 = all
X_train['RM'].values.reshape(-1,1)[:5]

