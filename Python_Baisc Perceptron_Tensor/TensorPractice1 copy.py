import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

x = np.array(3)
print(x) # 3
print(x.shape) # ()
print(np.ndim(x)) # 0

print('----------------------')
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
c = a+b
print(c) # [ 6  8 10 12]
print(c.shape) # (4,)
print(np.ndim(c)) # 1

print('----------------------')
c = a*b
print(c) # [ 5 12 21 32]
print(c.shape) # (4,)
print(np.ndim(c)) # 1

# Scalar * Vector
print('----------------------')
a = np.array(10) # Scalar
b = np.array([1,2,3,4]) # 1 Dimention Tensor
c = a*b
print(c) # [10 20 30 40]
print(c.shape) # (4,)
print(np.ndim(c)) # 1

#Transposed Vector
print('----------------------')
A = np.array([[1,2,3],[4,5,6]])
print(A) # [1 2 3]
         # [4 5 6]
print(A.shape) # (2, 3)
print(np.ndim(A)) # 2

print('----------------------')
A_ = A.T