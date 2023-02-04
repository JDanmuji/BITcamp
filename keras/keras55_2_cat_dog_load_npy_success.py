import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout

train_set_dir = 'C:/_data/dogs-vs-cats/train/train_set'
valid_set_dir = 'C:/_data/dogs-vs-cats/train/valid_set'
test_set_dir = 'C:/_data/dogs-vs-cats/train/test_set'


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)
 
train_generator = train_datagen.flow_from_directory(
    train_set_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_set_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_set_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)
 
train_step = train_generator.n // 32
valid_step = valid_generator.n // 32
test_step = test_generator.n // 32

model=Sequential()
model.add(Conv2D(kernel_size=(3,3),filters=3,input_shape=(150, 150, 1),activation="relu"))
model.add(Conv2D(kernel_size=(3,3),filters=10,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(kernel_size=(3,3),filters=3,activation="relu"))
model.add(Conv2D(kernel_size=(5,5),filters=5,activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(kernel_size=(2,2),strides=(2,2),filters=10))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adadelta",loss="binary_crossentropy",metrics=["accuracy"])
 
model.summary()

model.compile(optimizer='adam',
             loss='binary_crossentropy',
              metrics=['acc'])
              
model.fit_generator(train_generator,
                   steps_per_epoch=train_step,
                   epochs=10,
                   validation_data=valid_generator,
                   validation_steps=valid_step)

test_loss, test_acc = model.evaluate_generator(test_generator,
                                               steps=test_step,
                                               workers=4)


print(test_loss)
print(test_acc)

model.summary()

path = './_save/'
# path = '../_save/'
# path = 'C:/study/_save/' #절대경로

model.save(path + 'cat_dog_moedl.h5') 

# 0.5703628063201904
# 0.7066532373428345