from keras.datasets import mnist
from keras import models
from keras import layers

import warnings

warnings.filterwarnings('ignore')

(train_images, train_labels),(test_images,test_labels) = mnist.load_data()

print(train_images.shape) # (60000, 28, 28)
print(len(train_labels)) # 60000
print(test_images.shape) # (10000, 28, 28)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 데이터 타입의 변환
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 분류형 데이터의 설정
from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc: ', test_acc)

# Epoch 1/5
# 469/469 [==============================] - 4s 6ms/step - loss: 0.2568 - accuracy: 0.9262
# Epoch 2/5
# 469/469 [==============================] - 3s 7ms/step - loss: 0.1047 - accuracy: 0.9686
# Epoch 3/5
# 469/469 [==============================] - 3s 7ms/step - loss: 0.0693 - accuracy: 0.9785
# Epoch 4/5
# 469/469 [==============================] - 3s 7ms/step - loss: 0.0501 - accuracy: 0.9849
# Epoch 5/5
# 469/469 [==============================] - 3s 7ms/step - loss: 0.0380 - accuracy: 0.9889
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0717 - accuracy: 0.9777
# test_acc:  0.9776999950408936