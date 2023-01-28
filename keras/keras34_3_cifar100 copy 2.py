import numpy as np
import datetime


from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #파이썬 클래스 대문자로 시작   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


# 칼라
# 완성 후, 이메일 전송

# 100,
# 10

path = './_save/'


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data() #교육용 자료, 이미 train/test 분류

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)(훈련)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1) (테스트)

print(np.unique(y_train, return_counts=True))
#(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],dtype=int64))
      
   
   
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/ 255
x_test = x_test/ 255

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
  
   

# 2. 모델 구성 
model = Sequential()
model.add(Conv2D(filters=784, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu')) # (27, 27, 128)
model.add(MaxPooling2D())
model.add(Conv2D(filters=512, kernel_size=(2, 2), activation='relu'))                             # (26, 26, 64)
model.add(MaxPooling2D())
model.add(Conv2D(filters=256, kernel_size=(2, 2), activation='relu'))                             # (25, 25, 64)
model.add(Flatten())
model.add(Dense(128, activation='relu')) # input_shape=(60000, 40000) 6만 4만 인풋이야
model.add(Dense(100, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc', 'mae', 'mse'])

                                   
es = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=10,
                              restore_best_weights=True, 
                              verbose=1
                              )


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf

# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'min',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k34_3_cifer100' + date + '_'+ filename)



model.fit(x_train, y_train, epochs=100, batch_size=256, verbose=1, validation_split=0.25, callbacks=[es, mcp])

model.save(path + 'keras34_3_cifer100.h5')  #모델 저장 (가중치 포함 안됨)

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ',  result[0]) # loss
print('acc : ',  result[1]) # acc



'''
[튜닝 전]
loss :  4.605200290679932
acc :  0.009999999776482582

[데이터 전처리, float형태 변환, OneHotEncoding]
loss :  0.009900031611323357
acc :  0.009999999776482582

[MaxPooling 사용]
loss :  2.620914936065674
acc :  0.3562999963760376

acc :  0.3562999963760376

'''