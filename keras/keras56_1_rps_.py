import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize

path = 'C:/_data/rps/'


full_rock_name = os.listdir(path + 'rock')
labels = [each.split('0')[0] for each in full_rock_name]

full_scissors_name = os.listdir(path + 'scissors')
labels += [each.split('0')[0] for each in full_scissors_name]

full_paper_name = os.listdir(path + 'paper')
labels += [each.split('0')[0] for each in full_paper_name]

print(set(labels))

print(labels)

#(225, 299, 3)       
#(375, 499, 3)  

full_name = os.listdir(path)

from skimage.color import rgb2gray
import numpy as np

images = []
images1 = []
images2 = []
images3 = []

bar_total = full_rock_name
for file in bar_total:
    image = mpimg.imread(path + 'rock/' + file)
    images1.append(resize(image, (128, 128, 3)))
images = np.array(images1)

bar_total = full_scissors_name
for file in bar_total:
    image = mpimg.imread(path + 'scissors/' + file)
    images2.append(resize(image, (128, 128, 3)))
images += np.array(images2)


bar_total = full_paper_name
for file in bar_total:
    image = mpimg.imread(path+ 'paper/' + file)
    images3.append(resize(image, (128, 128, 3)))
images += np.array(images3)


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(labels)
labels_encoded = encoder.transform(labels)
labels_encoded[:3], encoder.classes_

from sklearn.model_selection import train_test_split

print(images.shape)
print(labels_encoded.shape)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels_encoded, test_size = 0.2, random_state = 13, stratify = labels_encoded
)

X_train.shape, X_test.shape


samples = random.choices(population = range(0,20000),k=8)

plt.figure(figsize = (14,12))
for idx, n in enumerate(samples):
    plt.subplot(2,4, idx+1)
    plt.imshow(X_train[n], cmap = 'Greys', interpolation = 'nearest')
    plt.title(y_train[n])

plt.tight_layout()
plt.show()


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
    layers.Dense(3, activation='softmax')
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


model.save(path + 'cat_dog_model1.h5')  