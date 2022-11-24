from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import os
from os.path import join

digits = load_digits()


plt.style.use('dark_background')

# See the data info
# print(digits.DESCR)
# Optical recognition of handwritten digits dataset
# --------------------------------------------------
# **Data Set Characteristics:**

#     :Number of Instances: 1797
#     :Number of Attributes: 64
#     :Attribute Information: 8x8 image of integer pixels in the range 0..16.
#     :Missing Attribute Values: None
#     :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
#     :Date: July; 1998
#     ...

#1data = 8x8x16bit(for color set)
data = digits.data

label = digits.target

#Check how big data is
# print(data.shape)
# (1797, 64)
# print(label.shape)
# (1797,)

# Data Formatting
# Step 1. check the data
# print(data[0])
# [ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
#  15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
#   0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
#   0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]

# Step2. reshape
# print(data[0].reshape(8,8))
# [[ 0.  0.  5. 13.  9.  1.  0.  0.]
#  [ 0.  0. 13. 15. 10. 15.  5.  0.]
#  [ 0.  3. 15.  2.  0. 11.  8.  0.]
#  [ 0.  4. 12.  0.  0.  8.  8.  0.]
#  [ 0.  5.  8.  0.  0.  9.  8.  0.]
#  [ 0.  4. 11.  0.  1. 12.  7.  0.]
#  [ 0.  2. 14.  5. 10. 12.  0.  0.]
#  [ 0.  0.  6. 13. 10.  0.  0.  0.]]
fig = plt.figure()
plt.imshow(data[0].reshape((8,8)))
print('Label: {}'.format(label[0]))

fig.savefig('my_figure.png')
plt.show()