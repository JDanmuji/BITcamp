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
# enc = OneHotEncoder(sparse=True)
# y = y.reshape(-1, 1)
# enc.fit(y)

# # sparse = True
# y = enc.transform(y).toarray()
# sparse = False
#y = enc.transform(y)

#ohe = OneHotEncoder()
#y = ohe.fit_transform(y)
#print(type(y))
print(y.shape) # 벡터형태, 1차원
# 데이터의 내용과 순서는 절대 바뀌게 하지 말아야된다.
y = y.reshape(581012, 1) # 형태 변환 
print(y.shape)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder() # 사이킷런 OneHotEncoder를 사용하면 toarray를 해줘야한다. (error :  Use X.toarray() to convert to a dense numpy array.)
#ohe.fit(y)  # 가중치 사용, 훈련할 때 영향을 끼침
#y = ohe.transform(y)            # (0, 4)        1.0
                                # (1, 4)        1.0
                                
y = ohe.fit_transform(y) 
print(y[:15])
print(type(y))
print(y.shape)
y = y.toarray()



x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    shuffle=True, 
    random_state=333, 
    test_size=0.2,
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
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.4,
          
          verbose=1)



                                                              
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

train : test : validation

[제일 이상적인 비율]
6 : 2 : 2
loss :  0.6624932885169983
accuracy :  0.7132776379585266
y_predict(예측값) [1 0 1 ... 1 1 1]
y_test(원래값) [1 6 4 ... 1 1 0]
0.7132776262230751


6 : 3
validation X
loss :  0.7118012309074402
accuracy :  0.6904488801956177
y_predict(예측값) [1 0 2 ... 1 1 1]
y_test(원래값) [1 1 5 ... 1 4 1]
0.6904488709381311


5 : 2.5 : 2.5
loss :  0.7331979870796204
accuracy :  0.6895623207092285
y_predict(예측값) [1 2 1 ... 1 0 1]
y_test(원래값) [1 3 1 ... 0 1 0]
0.6895623498309845


8 : 1 : 1
loss :  0.7144711017608643
accuracy :  0.6922308802604675
y_predict(예측값) [1 0 0 ... 1 0 1]
y_test(원래값) [0 0 0 ... 1 0 0]
0.6922309042717979


4 : 4 : 2
loss :  0.7611240148544312
accuracy :  0.6859189867973328
y_predict(예측값) [0 1 0 ... 1 2 0]
y_test(원래값) [0 1 1 ... 0 5 0]
0.6859189776467804


4 : 2 : 4
loss :  0.7693564891815186
accuracy :  0.6788551211357117
y_predict(예측값) [1 0 1 ... 1 1 1]
y_test(원래값) [1 6 4 ... 1 1 0]
0.6788551070110066

'''