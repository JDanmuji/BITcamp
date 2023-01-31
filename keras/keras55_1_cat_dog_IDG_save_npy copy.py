import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

print(os.listdir("C:/_data/dogs-vs-cats/"))

FAST_RUN = True
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

filenames = os.listdir("C:/_data/dogs-vs-cats/train/train/")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)


df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 

print(df)
print(df["category"])

# train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
# train_df = train_df.reset_index(drop=True)
# validate_df = validate_df.reset_index(drop=True)


df['category'].value_counts().plot.bar()
#validate_df['category'].value_counts().plot.bar()

total_df = df.shape[0]
#total_validate = validate_df.shape[0]
batch_size=15



train_datagen = ImageDataGenerator(
    rescale=1./255,
   
)

print(train_datagen)

train_datagen = train_datagen.flow_from_dataframe(
    df, 
    "C:/_data/dogs-vs-cats/train/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


print(train_datagen)
print(train_datagen[0][0])
print(train_datagen[0][1])

np.save('C:/_data/dogs-vs-cats/train/cats/train.npy', arr=train_datagen[0][0])
np.save('C:/_data/dogs-vs-cats/train/dogs/train.npy', arr=train_datagen[0][1])
# #np.save('./_data/brain/brain_xy_train.npy', arr=xy_train[0])

# np.save('E:/_data/dogs-vs-cats/dogs_vs_cats_x_test.py', arr=xy_test[0][0])
# np.save('E:/_data/dogs-vs-cats/dogs_vs_cats_y_test.py', arr=xy_test[0][1])


