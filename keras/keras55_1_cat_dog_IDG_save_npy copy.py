import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize

path = 'C:/_data/dogs-vs-cats1/train/'
os.listdir(path)[:5]

full_name = os.listdir(path)
labels = [each.split('.')[0] for each in full_name]
file_id = [each.split('.')[1] for each in full_name]

print(set(labels), len(file_id))

sample = random.choice(full_name)
image = mpimg.imread(path + sample)
print(image.shape)
sample = random.choice(full_name)
image = mpimg.imread(path + sample)
print(image.shape)

#(225, 299, 3)       
#(375, 499, 3)  

from skimage.color import rgb2gray
import numpy as np

images = []
bar_total = full_name
for file in bar_total:
    image = mpimg.imread(path + file)
    images.append(resize(image, (128, 128, 3)))
images = np.array(images)


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(labels)
labels_encoded = encoder.transform(labels)
labels_encoded[:3], encoder.classes_

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size = 0.2, random_state = 13, stratify = labels_encoded)

X_train.shape, X_test.shape

from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape = (128, 128, 3)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(2, activation='softmax')
])

model.summary()

import time
model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


start_time = time.time()
hist = model.fit(X_train.reshape(20000, 128, 128, 3), y_train, epochs = 5, verbose=1, validation_data= (X_test.reshape(5000, 128,128,3), y_test))

print(f'Fit Time : {time.time() - start_time}')

score = model.evaluate(X_test, y_test)
print(f'Test Loss : {score[0]}')
print(f'Test Accuracy  : {score[1]}')

path = './_save/'


model.save(path + 'cat_dog_model1.h5') 


import cv2
import matplotlib.pyplot as plt

image_bgr = cv2.imread("C:/Users/bitcamp/Desktop/dog/cat.jpg")
test_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
test_image = resize(test_image, (128, 128, 3))
plt.imshow(test_image)
plt.title('Cat')
plt.show()

if np.argmax(model.predict(test_image.reshape(1, 128, 128, 3))) == 0:
    print('Cat')
else :
    print('Dog')