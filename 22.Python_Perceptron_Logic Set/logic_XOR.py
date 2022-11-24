import numpy as np
import matplotlib.pyplot as plt

# XOR Gate
def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)

  return y

# NAND Gate
def NAND(a, b):
    input = np.array([a,b])
    
    #가중치 설정 (Weight)
    weights = np.array([-0.6,-0.6])
    bias = 0.7

    #출력값 (Output)
    value = np.sum(input * weights) + bias

    #반환값 (Return Value)
    if value <= 0:
        return 0
    else:
        return 1
    
# AND Gate    
def AND(a, b):
    input = np.array([a,b])
    
    #가중치 설정 (Weight)
    weights = np.array([0.4,0.4])
    bias = -0.6

    #출력값 (Output)
    value = np.sum(input * weights) + bias

    #반환값 (Return Value)
    if value <= 0:
        return 0
    else:
        return 1    
    
# OR gate    
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
    
    
    
print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))

