import numpy as np
import pandas as pd
import sklearn
import matplotlib as plt 
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
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
# print(data)
#    Sex  Length  Diameter  Height  Whole weight  Shucked weight  Viscera weight  Shell weight  Rings
# 0      M   0.455     0.365   0.095        0.5140          0.2245          0.1010        0.1500     15
# 1      M   0.350     0.265   0.090        0.2255          0.0995          0.0485        0.0700      7
# 2      F   0.530     0.420   0.135        0.6770          0.2565          0.1415        0.2100      9
# 3      M   0.440     0.365   0.125        0.5160          0.2155          0.1140        0.1550     10
# 4      I   0.330     0.255   0.080        0.2050          0.0895          0.0395        0.0550      7
# ...   ..     ...       ...     ...           ...             ...             ...           ...    ...
# 4172   F   0.565     0.450   0.165        0.8870          0.3700          0.2390        0.2490     11
# 4173   M   0.590     0.440   0.135        0.9660          0.4390          0.2145        0.2605     10
# 4174   M   0.600     0.475   0.205        1.1760          0.5255          0.2875        0.3080      9
# 4175   F   0.625     0.485   0.150        1.0945          0.5310          0.2610        0.2960     10
# 4176   M   0.710     0.555   0.195        1.9485          0.9455          0.3765        0.4950     12

# [4177 rows x 9 columns]

label = data['Sex']
del data['Sex']
# print(data.describe())
#             Length     Diameter       Height  Whole weight  Shucked weight  Viscera weight  Shell weight        Rings
# count  4177.000000  4177.000000  4177.000000   4177.000000     4177.000000     4177.000000   4177.000000  4177.000000
# mean      0.523992     0.407881     0.139516      0.828742        0.359367        0.180594      0.238831     9.933684
# std       0.120093     0.099240     0.041827      0.490389        0.221963        0.109614      0.139203     3.224169
# min       0.075000     0.055000     0.000000      0.002000        0.001000        0.000500      0.001500     1.000000
# 25%       0.450000     0.350000     0.115000      0.441500        0.186000        0.093500      0.130000     8.000000
# 50%       0.545000     0.425000     0.140000      0.799500        0.336000        0.171000      0.234000     9.000000
# 75%       0.615000     0.480000     0.165000      1.153000        0.502000        0.253000      0.329000    11.000000
# max       0.815000     0.650000     1.130000      2.825500        1.488000        0.760000      1.005000    29.000000

# print(data.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4177 entries, 0 to 4176
# Data columns (total 8 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   Length          4177 non-null   float64
#  1   Diameter        4177 non-null   float64
#  2   Height          4177 non-null   float64
#  3   Whole weight    4177 non-null   float64
#  4   Shucked weight  4177 non-null   float64
#  5   Viscera weight  4177 non-null   float64
#  6   Shell weight    4177 non-null   float64
#  7   Rings           4177 non-null   int64
# dtypes: float64(7), int64(1)
# memory usage: 261.2 KB
# None

#Sampling starts

#Random Over could polarize the data even more
ros = RandomOverSampler()
#Random Under could loss most critical data set
rus = RandomUnderSampler()

oversampled_data, oversampled_label = ros.fit_resample(data,label)
undersampled_data, undersampled_label = rus.fit_resample(data,label)

oversampled_data= pd.DataFrame(oversampled_data, columns=data.columns)
undersampled_data= pd.DataFrame(undersampled_data, columns=data.columns)

#Check the original Ratio
print('Original Data Class Ratio:\n{}'.format(pd.get_dummies(label).sum()))
# Original Data Class Ratio:
# F    1307
# I    1342
# M    1528
# dtype: int64

#Check the Oversampled Data Class Ratio
print('Oversample Data Class Ratio:\n{}'.format(pd.get_dummies(oversampled_label).sum()))
# Oversample Data Class Ratio:
# F    1528
# I    1528
# M    1528
# dtype: int64

#Check the Undersample Data Class Ratio
print('Undersample Data Class Ratio:\n{}'.format(pd.get_dummies(undersampled_label).sum()))
# Undersample Data Class Ratio:
# F    1307
# I    1307
# M    1307
# dtype: int64