import numpy as np
import pandas as pd
import tensorflow as tf
import datetime

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint     

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score



path = './_save/'

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets['target']


# One-hot Encoding 방법
# 1. keras 메서드 활용
y = to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    shuffle=True, 
    random_state=123,
    test_size=0.3,
    stratify=y
)



#2. 모델구성 
input1 = Input(shape=(4, ))
dense1 = Dense(50, activation='relu') (input1)
drop1 = Dropout(0.5) (dense1)
dense2 = Dense(40, activation='sigmoid') (drop1)
drop2 = Dropout(0.5) (dense2)
dense3 = Dense(30, activation='relu') (drop2)
drop3 = Dropout(0.5) (dense3)
dense4 = Dense(20, activation='linear') (drop3)
drop4 = Dropout(0.5) (dense4)
output1 = Dense(3, activation='softmax') (drop4)

model = Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                                  

es = EarlyStopping(monitor = 'val_loss', 
                   mode = 'min', 
                   patience = 20, #참을성     
                   #restore_best_weights = False, 
                   verbose = 1)



date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf


# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k31_07_' + date + '_'+ filename)

model.fit(x_train, 
          y_train, 
          epochs=500, 
          batch_size=8, 
          validation_split=0.25, 
          callbacks=[es, mcp])

model.save(path + 'keras31_dropout07_save_model.h5')  #모델 저장 (가중치 포함 안됨)


                                                              
#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)


y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1) # y_predict 가장 큰 값의 자릿수 뽑음 : 예측한 값

print( 'y_predict(예측값)' , y_predict)

y_test = np.argmax(y_test, axis=1) 

print( 'y_test(원래값)' , y_test)

acc = accuracy_score(y_test, y_predict) 

print(acc)

'''
loss :  0.02484283410012722
accuracy :  0.057244524359703064
y_predict(예측값) [2 0 1 1 0 2 2 2 2 2 0 2 0 2 1 2 0 0 2 2 2 2 2 1 2 1 0 2 1 0 0 0 0 1 1 1 0
 1 0 2 0 1 1 0 1]
y_test(원래값) [1 0 1 1 0 2 2 1 2 2 0 2 0 2 1 2 0 0 2 2 2 2 2 1 2 1 0 2 1 0 0 0 0 1 1 1 0
 1 0 2 0 1 1 0 1]
0.9555555555555556


'''