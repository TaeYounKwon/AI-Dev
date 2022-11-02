import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras import models
from keras import optimizers

# Image Scaling
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img






# Original Data Path
original_dataset_dir = 'datasets/train'

# Small Dataset Path
base_dir = 'datasets/cats_and_dogs_small'

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.mkdir(base_dir)    

# Set up all the data paths
# Folder Creation(Train, Validation, Test Data)
train_dir = os.path.join(base_dir,'train')
os.mkdir(train_dir)

validation_dir = os.path.join(base_dir,'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)

# For Cats and Dogs
train_cats_dir = os.path.join(train_dir,'cats')
train_dogs_dir = os.path.join(train_dir,'dogs')

os.mkdir(train_cats_dir)
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir,'cats')
validation_dogs_dir = os.path.join(validation_dir,'dogs')
os.mkdir(validation_cats_dir)
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir,'cats')
test_dogs_dir = os.path.join(test_dir,'dogs')
os.mkdir(test_cats_dir)
os.mkdir(test_dogs_dir)

# File Copy
# fnames = []
# for i in range(1000):
#     filename = 'cat.{}.jpg'.format(i)
#     fnames.append(filename)

# cat train data copy
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)
    
# dog train data copy
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)
print('----------------------------  Train dataset copy completed')    
    
    
# cat validation data copy
fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src,dst)    
    
# dog validation data copy
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_dogs_dir,fname)
    shutil.copyfile(src,dst)        
print('----------------------------  Validation dataset copy completed')

# cat test data copy
fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)    
    
# dog test data copy
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)        
print('----------------------------  Test dataset copy completed')        

print('Train cat images: ', len(os.listdir(train_cats_dir)))
print('Train dog images: ', len(os.listdir(train_dogs_dir)))

print('Validation cat images: ', len(os.listdir(validation_cats_dir)))
print('Validation dog images: ', len(os.listdir(validation_dogs_dir)))

print('Test cat images: ', len(os.listdir(test_cats_dir)))
print('Test dog images: ', len(os.listdir(test_dogs_dir)))

# Build network

model =models.Sequential()
# (#,#,3) for color, (#,#,1) for black and white , #=real pixcel x&y
model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape = (150,150,3))) 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))              
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))              
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2))) 
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

print('----------------------------  Model Summary')   
print(model.summary())

model.compile(optimizer='rmsprop',
              loss ='binary_crossentropy',
              metrics=['accuracy'])

# Data Processing (데이터 전처리 & 데이터 개수 증가)
# Image Scaling (이미지 크기 조정(모두 같은 크기로))

# 1./255 = ratio
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                                  train_dir,
                                  target_size=(150,150),
                                  batch_size=20,
                                  class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
                                  validation_dir,
                                  target_size=(150,150),
                                  batch_size=20,
                                  class_mode='binary')

# Check the Resized data
for data_batch, labels_batch in train_generator:
    print('Batch Data Size: ', data_batch.shape)
    print('Batch Label Size: ', labels_batch.shape)
    break

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data= validation_generator,
    validation_steps=50
)
model.save('cats_and_dogs_small_1.0.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label = 'Training Accuracy')
plt.plot(epochs,val_acc,'b-',label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('Training Accuracy.png')

plt.figure()
plt.plot(epochs,loss,'bo',label = 'Training Loss')
plt.plot(epochs,val_loss,'b-',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('Training loss.png')


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'    
)

fnames = sorted([os.path.join(train_cats_dir,fname)  for fname in os.listdir(train_cats_dir)])
img_path = fnames[4]
img = load_img(img_path, target_size=(150,150))

x = tf.keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,)+x.shape)
i=0
for batch in datagen.flow(x,batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i% 4 ==0:
        break
plt.show()

train_datagen = ImageDataGenerator(    
                                   rescale=1./255,    
                                   rotation_range=40,    
                                   width_shift_range=0.2,    
                                   height_shift_range=0.2,    
                                   shear_range=0.2,    
                                   zoom_range=0.2,    
                                   horizontal_flip=True,)

# 검증 데이터는 증식되어서는 안 됩니다!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    # 타깃 디렉터리        
    train_dir,        
    # 모든 이미지를 150 × 150 크기로 바꿉니다        
    target_size=(150, 150),        
    batch_size=32,        
    # binary_crossentropy 손실을 사용하기 때문에 이진 레이블을 만들어야 합니다        
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(        
                                                        validation_dir,        
                                                        target_size=(150, 150),        
                                                        batch_size=32,        
                                                        class_mode='binary')

history = model.fit_generator(      
                              train_generator,      
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)