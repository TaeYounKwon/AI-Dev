import numpy as np
import pandas as pd
import sklearn
import matplotlib as plt 
import os
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

# Min-Max Data Scaling, all the values are in the same range limit(0~1)
# Check the difference between first data[Rings] and second data[Rings]
data = (data-np.min(data)) / (np.max(data) - np.min(data))
# print(data)
#         Length  Diameter    Height  Whole weight  Shucked weight  Viscera weight  Shell weight     Rings
# 0     0.513514  0.521008  0.084071      0.181335        0.150303        0.132324      0.147982  0.500000
# 1     0.371622  0.352941  0.079646      0.079157        0.066241        0.063199      0.068261  0.214286
# 2     0.614865  0.613445  0.119469      0.239065        0.171822        0.185648      0.207773  0.285714
# 3     0.493243  0.521008  0.110619      0.182044        0.144250        0.149440      0.152965  0.321429
# 4     0.344595  0.336134  0.070796      0.071897        0.059516        0.051350      0.053313  0.214286
# ...        ...       ...       ...           ...             ...             ...           ...       ...
# 4172  0.662162  0.663866  0.146018      0.313441        0.248151        0.314022      0.246637  0.357143
# 4173  0.695946  0.647059  0.119469      0.341420        0.294553        0.281764      0.258097  0.321429
# 4174  0.709459  0.705882  0.181416      0.415796        0.352724        0.377880      0.305431  0.285714
# 4175  0.743243  0.722689  0.132743      0.386931        0.356422        0.342989      0.293473  0.321429
# 4176  0.858108  0.840336  0.172566      0.689393        0.635171        0.495063      0.491779  0.392857

# [4177 rows x 8 columns]