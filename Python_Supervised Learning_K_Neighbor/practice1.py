from sklearn.datasets import load_iris

iris_dataset = load_iris()

# 1. Check data info, type, structure, 
# print(iris_dataset)

# print('Key of iris_dataset:\n',iris_dataset.keys())
# Key of iris_dataset: 
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

# print('\nTarget Name:\n',iris_dataset['target_names'])
# Target Name:
# ['setosa' 'versicolor' 'virginica']

# print('Data Type:\n',type(iris_dataset['data']))
# Data Type:
# <class 'numpy.ndarray'>

# print('Data Size:\n',iris_dataset['data'].shape)
# Data Size:
# (150, 4)

# print('First 5 Data:\n', iris_dataset['data'][:5])
# First 5 Data:
#  [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]

# print('Target Type: ', type(iris_dataset['target']))
# Target Type:  <class 'numpy.ndarray'>

# print('Target Type: ', iris_dataset['target'].shape)
# Target Type:  (150,)

# print('target:\n',iris_dataset['target'])
# target:
#  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]

# 2. Getting Read to use Data
# Divide the data to train, test(Both X & Y)
from sklearn.model_selection import train_test_split
X_train, X_Test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],random_state=0, test_size=.278)
# print('X_train Size: ', X_train.shape)
# print('X_Test Size: ', X_Test.shape)
# print('Y_train Size: ', Y_train.shape)
# print('Y_test Size: ',Y_test.shape)
# X_train Size:  (108, 4)
# X_Test Size:  (42, 4)
# Y_train Size:  (108,)
# Y_test Size:  (42,)


# 3. Data visualization
import pandas as pd
import matplotlib.pyplot as plt


iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])
# print(iris_dataframe)
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                  6.9               3.1                5.1               2.3
# 1                  5.0               3.5                1.6               0.6
# 2                  5.4               3.7                1.5               0.2
# 3                  5.0               2.0                3.5               1.0
# 4                  6.5               3.0                5.5               1.8
# ..                 ...               ...                ...               ...
# 103                4.9               3.1                1.5               0.1
# 104                6.3               2.9                5.6               1.8
# 105                5.8               2.7                4.1               1.0
# 106                7.7               3.8                6.7               2.2
# 107                4.6               3.2                1.4               0.2p

# [108 rows x 4 columns]
pd.plotting.scatter_matrix(iris_dataframe, figsize=(15,15), c=Y_train, marker='o',alpha=0.8)


plt.style.use('dark_background')

plt.savefig('my_figure.png')
#plt.show()

# 4. 1st Machine Learning Model: KNeighbors algorithm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)

#Predict this new value
X_new = np.array([[5., 2.9, 1., 0.2]])
prediction = knn.predict(X_new)
# print('Predict Value: ', prediction)
# print('Predict Target Name: ', iris_dataset['target_names'][prediction])
# Predict Value:  [0]
# Predict Target Name:  ['setosa']

y_pred = knn.predict(X_Test)
# print('Predict Test Set Value:\n',y_pred)
#  [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  2 1 1 2 0]

# 5. Check machihne learning prediction rate
# Check how y_pred values are smiliar to y_test
PredctionRate = np.mean(y_pred == Y_test)*100
# print('Prediction Rate: ',round(PredctionRate,2),'%')
# Prediction Rate:  97.62 %

# OR

print('Prediction Rate: ',round(knn.score(X_Test,Y_test)*100,2),'%')
