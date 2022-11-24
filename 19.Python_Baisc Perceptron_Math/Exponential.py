import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# y= ax
def exponential_function(x):
    a= 4
    
    return a**x 

x = np.arange(-3,2,0.1)
y = exponential_function(x)

plt.plot(x,y)
plt.xlabel('x-Value')
plt.ylabel('y-Value')
plt.xlim(-4,3)
plt.ylim(-1,15)
plt.title('Exponential Function Graph')
plt.savefig('Exponential_Graph.png')
plt.show()