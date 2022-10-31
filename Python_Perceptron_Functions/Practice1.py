import numpy as np
import matplotlib.pyplot as plt


# Step Function
def step_function(x):
    if x>0:
        return 1
    else:
        return 0

# For numpy y value
def step_function_for_numpy(x):
    y=x>0
    value = y.astype(np.int)
    return value    

#Sigmoid Function
def sigmoid(x):
  value = 1 / (1 + np.exp(-x))
  return value

# ReLu(x)
def ReLu(x):
    if x>0:
        return X
    else:
        return 0
    
# Identity Function(항등함수)
def identity_function(x):
    return x    

print("Step Function")
print(step_function(-3))
print(step_function(5))    

print("-----------------------")
print("Sigmoid Function")
print(sigmoid(3))
print(sigmoid(-3))

plt.grid()
x = np.arange(-5,5,0.01)
y1 = sigmoid(x)
y2 = step_function_for_numpy(x)
plt.plot(x, y1,'r-')
plt.plot(x,y2,'b--')
plt.savefig('function graphs.png')
plt.show()