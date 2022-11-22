import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# y= ax^2 + bx +c
def quadratic_function(x):
    a= 1
    b =-3
    c = 10
    return a*x**2 +b*x +c

x = np.arange(-10,10,0.1)
y = quadratic_function(x)

plt.plot(x,y)
plt.scatter(1.5,quadratic_function(1.5))
plt.xlabel('x-Value')
plt.ylabel('y-Value')
plt.title('Quadratic Function Graph')
plt.savefig('Quadratic_Custom.png')
plt.show()