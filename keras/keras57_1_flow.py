import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size=100



xy_train = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest' 
)

xy_test= ImageDataGenerator(
      rescale=1./255
)
                        
x_data = xy_train.flow( # -1, 전체 데이터
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1),      # x    
    np.zeros(augument_size),                                                        # y
    batch_size=augument_size,
    shuffle=True,
) 
print(x_data[0])
print(x_data[0][0].shape)
print(x_data[0][1].shape)

                      
xy_test = xy_test.flow_from_directory(  # flow_from_directory : 데이터를 가져와 수치화 해준 뒤 작업해주는 거
    './_data/brain/test/', 
    target_size=(200,200), 
    batch_size=10, 
    class_mode='binary', 
    color_mode='grayscale',
    shuffle=True
    
) 

import matplotlib.pyplot as plt

plt.figure(figsize=(7,7))

for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()
    