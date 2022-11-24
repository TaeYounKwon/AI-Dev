import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# y= ax^3 + bx^2 +cx +d
def cubic_function(x):
    a= 4
    b =0
    c = -1
    d= -8
    
    return a*x**3 +b*x**2 +c*x+d

x = np.arange(-5,5,0.1)
y = cubic_function(x)

plt.plot(x,y)
plt.xlabel('x-Value')
plt.ylabel('y-Value')
plt.title('Cubic Function Graph')

plt.savefig('Cubic_Graph.png')
plt.show()