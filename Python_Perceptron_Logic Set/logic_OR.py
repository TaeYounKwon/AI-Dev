import numpy as np
import matplotlib.pyplot as plt

# OR Gate
def OR(a, b):
    input = np.array([a,b])
    
    #가중치 설정(Weight)
    weights = np.array([0.4,0.4])
    bias = -0.3

    #출력값 (Output)
    value = np.sum(input * weights) + bias

    #반환값 (Return Value)
    if value <= 0:
        return 0
    else:
        return 1
    
print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))

x1 = np.arange(-2,2, 0.01)
x2 = np.arange(-2,2, 0.01)
bias = -0.3

y = (-0.4 * x1 - bias) / 0.4

plt.plot(x1, y, 'r--')
plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)
plt.grid()
plt.scatter(0,0, color='orange',marker='o',s=150)
plt.scatter(0,1, color='red',marker='^',s=150)
plt.scatter(1,0, color='red',marker='^',s=150)
plt.scatter(1,1, color='red',marker='^',s=150)


plt.savefig('OR_grpah.png')
plt.show()