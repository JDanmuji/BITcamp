import numpy as np
import datetime


from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #파이썬 클래스 대문자로 시작   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

path = './_save/'


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data() #교육용 자료, 이미 train/test 분류

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델 구성 
model = Sequential()
model.add(Conv2D(filters=256, 
                 kernel_size=(3, 3), 
                 padding='same',
                 input_shape=(32, 32, 3), 
                 activation='relu')) # (27, 27, 128)
model.add(MaxPool2D())
model.add(Conv2D(filters=128, kernel_size=(2, 2)))                             # (26, 26, 64)
model.add(Conv2D(filters=64, kernel_size=(2, 2)))                             # (25, 25, 64)
model.add(Flatten())
model.add(Dense(32, activation='relu')) # input_shape=(60000, 40000) 6만 4만 인풋이야
model.add(Dense(24, activation='relu')) 
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
#sparse_categorical_crossentropy
model.compile(loss='mse', optimizer='adam', metrics=['acc', 'mae'])

                                   
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
[튜닝 전]
loss :  2.3026225566864014
acc :  0.10000000149011612

[튜닝 후]
loss :  0.0570102222263813
acc :  0.5834000110626221

loss :  0.05636516585946083
acc :  0.5734999775886536

[튜닝 후 + padding 추가]
loss :  0.05055766925215721
acc :  0.6243000030517578
'''