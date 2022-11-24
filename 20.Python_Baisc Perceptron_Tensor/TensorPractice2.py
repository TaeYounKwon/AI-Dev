import math
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-whitegrid')

from keras.datasets import mnist
# Get Data from keras

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
print(train_images.ndim) # 3
print(train_images.shape) # (60000, 28, 28)
print(train_images.dtype) # uint8

print(train_labels[0]) # 5

temp_image = train_images[0]

plt.imshow(temp_image, cmap = 'gray')
plt.show()