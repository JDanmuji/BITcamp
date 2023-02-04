import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout

# 이미지를 변환하고 증폭시키는 역할
xy_train = ImageDataGenerator(
    rescale=1./255, # 원본을 수치화 한 것만 가지고 있음
)

xy_test= ImageDataGenerator(
      rescale=1./255
)
                        #파일
xy_train = xy_train.flow_from_directory( # 안에 있는 ad, normal 은 0, 1로 인식
    'C:/_data/dogs-vs-cats/train/', 
    target_size=(200,200), #크기에 상관없이 200, 200 을 압축
    batch_size=100000, # 모든 데이터를 가지고 오기
    class_mode='categorical', #수치
    color_mode='grayscale',
    shuffle=True
    # Found 160 images belonging to 2 classes.
) 
# 

#                      #파일
# xy_test = xy_test.flow_from_directory( # 안에 있는 ad, normal 은 0, 1로 인식
#     'C:/_data/dogs-vs-cats/test1/', 
#     target_size=(200,200), #크기에 상관없이 200, 200 을 압축
#     batch_size=100000,  #batch_size를 최대한 늘려서 데이터를 한번에 뽑아낼 수 있음
#     class_mode='categorical', #수치
#     color_mode='grayscale',
#     shuffle=True
#     # Found 120 images belonging to 2 classes.
# ) 

print(xy_train)


print(xy_train[0][1])
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

# np.save('E:/_data/dogs-vs-cats/dogs_vs_cats_x_train.npy', arr=xy_train[0][0])
# np.save('E:/_data/dogs-vs-cats/dogs_vs_cats_y_train.npy', arr=xy_train[0][1])
# #np.save('./_data/brain/brain_xy_train.npy', arr=xy_train[0])

# np.save('E:/_data/dogs-vs-cats/dogs_vs_cats_x_test.py', arr=xy_test[0][0])
# np.save('E:/_data/dogs-vs-cats/dogs_vs_cats_y_test.py', arr=xy_test[0][1])



x_train = np.load('E:/_data/dogs-vs-cats/dogs_vs_cats_x_train.npy')
y_train = np.load('E:/_data/dogs-vs-cats/dogs_vs_cats_y_train.npy')
x_test = np.load('E:/_data/dogs-vs-cats/dogs_vs_cats_x_test.npy')
y_test = np.load('E:/_data/dogs-vs-cats/dogs_vs_cats_y_test.npy')

print(x_train.shape, x_test.shape) 
# binary
#(25000, 200, 200, 1) (25000, 1)

print(y_train.shape, y_test.shape) 
# binary
#(25000, 1) (12500, 1)

print(x_train[100])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(kernel_size=(3,3),filters=3,input_shape=(200,200,1),activation="relu"))
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


# 3. 컴파일, 훈련
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

                # 전체 x 값      # 전체 y 값
hist = model.fit(x_train, y_train,
                    epochs=50, 
                    validation_data=(x_test, y_test), 
                    validation_split=0.3,
                    batch_size=64,
                    )



accuracy = hist.history['acc'] 

val_acc = hist.history['val_acc'] 
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# [-1] 모든 훈련에 관한 loss 값이 나오기에 맨 마지막 훈련의 값을 출력 
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])
print('loss : ', loss[-1]) 
print('val_loss : ', val_loss[-1])


# 그림 그리삼



# batch = xy_train.next() # next 파이썬 for 문 내장함수

# print(batch)
# print(len(batch)) #2
# print(type(batch)) #tuple

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 10))

# for i in range(10) :
#     plt.imshow(batch[0][i])
# plt.show()
