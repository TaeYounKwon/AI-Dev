# Check all necessary Library   
import sklearn
from sklearn.preprocessing import *
import numpy as np
from numpy import * 

# 표준편차 구현 함수
def numpy_standardization(data):
    """ 
    (각데이터 - 평균(각열)) / 표준편차(각열)
    """
    std_data = (data-np.mean(data, axis=0)) / np.std(data, axis=0)
    return std_data

# 정규화 구현 함수
def normalization(data):
    data_mm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))     
    return data_mm

def main():
    data =np.random.randint(30, size=(6,5))
    # print(data)
    
    std_data_tmp = numpy_standardization(data)
    # print(std_data_tmp)

    data_mm_tmp = normalization(data)
    print(data_mm_tmp)
    
if __name__=='__main__':
    main()    
    