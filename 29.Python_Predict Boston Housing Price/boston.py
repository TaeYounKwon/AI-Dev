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

k = 4
num_epochs = 500

num_val_samples = len(train_data)//k # 폴더의 사이즈
all_scores = []

for i in range(k):
    print('Working on fold #',i)
    
    # 검증 데이터를 준비: k번째 분할
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_labels = train_labels[i*num_val_samples:(i+1)*num_val_samples]
    
    # 훈련 데이터를 준비:
    particial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]], axis=0)
    
    # 훈련 라벨 준비:
    particial_train_labels = np.concatenate([train_labels[:i*num_val_samples],train_labels[(i+1)*num_val_samples:]], axis=0)
    
    # print(i*num_val_samples,(i+1)*num_val_samples)
    
    model = build_model()
    history = model.fit(particial_train_data, 
                        particial_train_labels, 
                        epochs=num_epochs, 
                        batch_size=1, 
                        validation_data=(val_data, val_labels),
                        verbose=0)
    mae_history = history.history['mae']

    all_scores.append(mae_history)
    

# 각 epoch 별로 평균을 구해냄
average_mae_history = [np.mean([x[i] for x in all_scores]) for i in range(num_epochs)] 

# Data Visualization

plt.plot(range(1,len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')