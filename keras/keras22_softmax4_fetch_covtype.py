import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping #파이썬 클래스 대문자로 시작   
from sklearn.metrics import accuracy_score


import pandas as pd
import tensorflow as tf


# 1. 데이터 
datasets = fetch_covtype()


# 선생님질문 : x, y 데이터 뽑아올 때, 이미 array
x = datasets.data
y = datasets['target']


# 1. to_categorical() 사용
# y = to_categorical(y)
# y = np.delete(y,0,axis=1)

# 2. get_dummies() 사용
# y = pd.get_dummies(y, drop_first=False)
# print(type(y))
# 판다스 .value    .numpy()
# y = np.array(y)

# 3. OneHotEncoder() 사용
# type     
# sparse = True(Default), Mactrics 반환, toArray(가 필요)
# sparse = False, array 변환
enc = OneHotEncoder(sparse=True)
y = y.reshape(-1, 1)
enc.fit(y)

# sparse = True
y = enc.transform(y).toarray()
# sparse = False
#y = enc.transform(y)

#ohe = OneHotEncoder()
#y = ohe.fit_transform(y)
#print(type(y))



x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    shuffle=True, 
    random_state=333, 
    test_size=0.3,
    stratify=y 
)


#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(54,)))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(7, activation='softmax')) 


                                                                                          
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=2,
                              restore_best_weights=True, 
                              verbose=1
                              )

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)



                                                              
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

loss :  0.708001434803009
accuracy :  0.6980562806129456
y_predict(예측값) [1 0 2 ... 1 1 1]
y_test(원래값) [1 1 5 ... 1 4 1]
0.6980562695061502

'''