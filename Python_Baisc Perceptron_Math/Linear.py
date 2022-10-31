import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# y = ax +b
def linear_function(x):
    a= 0.5
    b =2
    return a*x +b

x = np.arange(-5,5,0.1)
y = linear_function(x)

plt.plot(x,y)
plt.xlabel('x-Value')
plt.ylabel('y-Value')
plt.title('Linear Function Graph')
plt.savefig('Linear_Graph.png')
plt.show()