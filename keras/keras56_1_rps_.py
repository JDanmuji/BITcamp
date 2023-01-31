import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
 
 
path_dir1 = 'C:/_data/rps/rock/'
path_dir2 = 'C:/_data/rps/paper/'
path_dir3 = 'C:/_data/rps/scissors/'
 
file_list1 = os.listdir(path_dir1) # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2)
file_list3 = os.listdir(path_dir3)
 
 
#%% train용 이미지 준비
num = 0;
train_img = np.float32(np.zeros((2520, 224, 224, 3))) # 
train_label = np.float64(np.zeros((2520, 1)))
 
print(len(file_list1))
print(len(file_list2))
print(len(file_list3))
print(len(train_img))
print(len(train_label))

 
for img_name in file_list1:
    img_path = path_dir1+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num + 1
 
for img_name in file_list2:
    img_path = path_dir2+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 1 # paper
    num = num + 1
 
for img_name in file_list3:
    img_path = path_dir3+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 2 # scissors
    num = num + 1
 
 
# 이미지 섞기
     
n_elem = train_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)
 
train_label = train_label[indices]
train_img = train_img[indices]
 
#%% 
# create the base pre-trained model
IMG_SHAPE = (224, 224, 3)
 
base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print("Number of layers in the base model: ", len(base_model.layers))
 
GAP_layer = GlobalAveragePooling2D()
dense_layer = Dense(3, activation=tf.nn.softmax)
 
model = Sequential([
        base_model,
        GAP_layer,
        dense_layer
        ])
 
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
 
path = './_save/'

model.fit(train_img, train_label, epochs=5)
 
# save model
model.save(path + 'rps_model.h5')  

 
print("Saved model to disk")  
 