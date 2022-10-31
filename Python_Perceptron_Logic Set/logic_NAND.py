import numpy as np
import matplotlib.pyplot as plt

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
    
print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))

