import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.model_selection import train_test_split

path = 'C:/_data/dogs-vs-cats1/train/'

full_name = os.listdir(path)
labels = [each.split('.')[0] for each in full_name]
file_id = [each.split('.')[1] for each in full_name]

print(set(labels), len(file_id))

images = []
bar_total = full_name
for file in bar_total:
    image = mpimg.imread(path + file)
    images.append(resize(image, (128, 128, 3)))
images = np.array(images)

encoder = LabelEncoder()
encoder.fit(labels)
labels_encoded = encoder.transform(labels)
labels_encoded[:3], encoder.classes_

print(images.shape)
print(labels_encoded.shape)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels_encoded, test_size=0.2, random_state = 13, stratify=labels_encoded
)

X_train.shape, X_test.shape

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape = (128, 128, 3)))
model.add(Conv2D)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(2, activation='softmax')    


model.summary()


model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
)

hist = model.fit(X_train.reshape(20000, 128, 128, 3), y_train, epochs = 5, verbose=1, validation_data= (X_test.reshape(5000, 128,128,3), y_test))


score = model.evaluate(X_test, y_test)

print(f'Test Loss : {score[0]}')
print(f'Test Accuracy  : {score[1]}')


model.save(path + 'cat_dog_model1.h5')  