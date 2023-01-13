import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #파이썬 클래스 대문자로 시작   
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import tensorflow as tf


path = './_save/'


# 1. 데이터 
datasets = fetch_covtype()

x = datasets.data
y = datasets['target']

y = y.reshape(581012, 1) 


ohe = OneHotEncoder() 
         
y = ohe.fit_transform(y) 
y = y.toarray()


x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    shuffle=True, 
    random_state=333, 
    test_size=0.3,
    stratify=y 
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
input1 = Input(shape=(54, ))
dense1 = Dense(50, activation='relu') (input1)
drop1 = Dropout(0.5) (dense1)
dense2 = Dense(40, activation='sigmoid') (drop1)
drop2 = Dropout(0.5) (dense2)
dense3 = Dense(30, activation='relu') (drop2)
drop3 = Dropout(0.5) (dense3)
dense4 = Dense(20, activation='linear') (drop3)
drop4 = Dropout(0.5) (dense4)
output1 = Dense(7, activation='softmax') (drop4)

model = Model(inputs=input1, outputs=output1)
model.summary() #Total params: 4,611

                                                   
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                                       
es = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=2,
                              restore_best_weights=True, 
                              verbose=1
                              )


import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
print(date)   #0112_1502
print(type(date)) #<class 'str'>


filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf

# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k31_10_' + date + '_'+ filename)

model.fit(x_train, 
          y_train, 
          epochs=5000, 
          batch_size=32, 
          validation_split=0.25, 
          callbacks=[es, mcp])

#model.save(path + 'keras31_dropout10_save_model.h5')  #모델 저장 (가중치 포함 안됨)



                                                              
#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)



y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)

print( 'y_predict(예측값)' , y_predict)

y_test = np.argmax(y_test, axis=1)

print( 'y_test(원래값)' , y_test)

acc = accuracy_score(y_test, y_predict) 

print(acc)





'''

loss :  0.6085023283958435
accuracy :  0.7451693415641785
y_predict(예측값) [1 0 2 ... 1 1 1]
y_test(원래값) [1 1 5 ... 1 4 1]
0.7451693592803378

'''