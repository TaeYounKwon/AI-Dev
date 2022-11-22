import keras
import numpy as np
import matplotlib.pyplot as plt

# 보스턴 주택 가격 데이터셋

from keras.datasets import boston_housing
(train_data,train_labels),(test_data,test_labels) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
#train_data = train_data - mean 와 같은 표현

# 표준 편차를 구한다. 
std = train_data.std(axis=0)

train_data /= std
#train_data = train_data / std

mean = test_data.mean(axis=0)
test_data -= mean
std = test_data.std(axis=0)
test_data /= std

# 신경망을 만든다. 
from keras import models
from keras import layers

def build_model():
  model = models.Sequential()
  model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1))

  model.compile(optimizer='rmsprop', 
                loss='mse',
                metrics=['mae'])
  return model