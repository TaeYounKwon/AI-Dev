import msilib
import numpy as np
import pandas as pd
import matplotlib as plt 
import os

from sklearn.preprocessing import LabelEncoder
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

le=LabelEncoder()
# print(type(label))
# <class 'pandas.core.series.Series'>
# print(label)
# 0       M
# 1       M
# 2       F
# 3       M
# 4       I
#        ..
# 4172    F
# 4173    M
# 4174    M
# 4175    F
# 4176    M
# Name: Sex, Length: 4177, dtype: object



# Change M,F,I to 2,0,1
label_encoded_label = le.fit_transform(label)
# print(label_encoded_label)
# [2 2 0 ... 2 0 2]