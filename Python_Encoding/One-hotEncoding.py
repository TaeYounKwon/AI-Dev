import msilib
import numpy as np
import pandas as pd
import matplotlib as plt 
import os

from sklearn.preprocessing import OneHotEncoder
from os.path import join

abalone_path = join('.','abalone.txt')
column_path = join('.','abalone_attributes.txt')
# print(abalone_path)
# .\abalone.txt

abalone_columns = list()
for line in open(column_path):
    abalone_columns.append(line.strip())
    
# print(abalone_columns) 
# ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

data = pd.read_csv(abalone_path, header=None, names=abalone_columns)
label = data['Sex']

# One-hot-Encoder
# if data is M(2), F(0), I(1) ...
# will produce [0,0,], [1,0,0],[0,1,0]


ohe = OneHotEncoder(sparse=False)
one_hot_encoded = ohe.fit_transform(label.values.reshape((-1,1)))
# print(one_hot_encoded)
# [[0. 0. 1.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  ...
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
