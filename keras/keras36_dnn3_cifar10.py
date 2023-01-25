import numpy as np
import datetime


from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #파이썬 클래스 대문자로 시작   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout

path = './_save/'


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data() #교육용 자료, 이미 train/test 분류

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')


x_train = x_train/255.
x_test = x_test/255.

#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

# 2. 모델 구성 
model = Sequential()
model.add(Flatten(input_shape=(32*32*3,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='linear'))
model.add(Dense(24, activation='linear'))
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
#sparse_categorical_crossentropy
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])

                                   
es = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=50,
                              restore_best_weights=True, 
                              verbose=1
                              )





date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf

# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k34_2_cifer10' + date + '_'+ filename)



model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.25, callbacks=[es, mcp])

model.save(path + 'keras34_2_cifer10.h5')  #모델 저장 (가중치 포함 안됨)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ',  result[0]) # loss
print('acc : ',  result[1]) # acc


'''
[dnn]
loss :  0.0570102222263813
acc :  0.5834000110626221

[cnn]
loss :  0.07736314088106155
acc :  0.3564999997615814

loss :  1.540274977684021
acc :  0.45080000162124634



'''

