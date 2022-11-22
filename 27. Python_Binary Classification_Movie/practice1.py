import keras
# print(keras.__version__) 2.10.0
from keras.datasets import imdb


(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)
# print(train_data.shape) (25000,)

word_index = imdb.get_word_index()
word_index.items()
reverse_word_index = dict([value, key] for (key, value) in word_index.items())

decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
# print(decoded_review)
# ? this film was just brilliant casting location scenery story direction everyone's really 
# suited the part they played and you could just imagine being there robert ? 
# is an amazing actor and now the same being director ? 
# father came from the same scottish island as myself so i loved the fact there was a real connection
# with this film the witty remarks throughout the film were great it was just brilliant so much that
# i bought the film as soon as it was released for ? and would recommend it to everyone to watch 
# and the fly fishing was amazing really cried at the end it was so sad and you know what they say 
# if you cry at a film it must have been good and this definitely was also ? to the two little boy's 
# that played the ? of norman and paul they were just brilliant children are often left out of the ? 
# list i think because the stars that play them all grown up are such a big profile for 
# the whole film but these children are amazing and should be praised for what they have done don't 
# you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all

# Getting Read to use Data
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences),dimension))

  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1

  return results

# Data의 Encoding
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Chnage the data type to float 
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Build neural Network
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from tensorflow.keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# Bring the test result data. 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

# Visualize the test result data. 
import matplotlib.pyplot as plt

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss,'b-', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Result.png')
plt.show()


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=3,
                    batch_size=256,
                    validation_data=(x_val, y_val))