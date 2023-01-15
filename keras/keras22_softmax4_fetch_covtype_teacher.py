import numpy as np
from sklearn.datasets import fetch_covtype


from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping #파이썬 클래스 대문자로 시작   
from sklearn.metrics import accuracy_score


import pandas as pd
import tensorflow as tf


# 1. 데이터 
datasets = fetch_covtype()

print(type(datasets))

x = datasets.data
y = datasets['target']

print(type(x))
print(type(y))

########################### 케라스 to_categorical컬 ###################################
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y.shape)  #(581012, 8) 
# # 0이 들어있는 필요없는 열이 생기게 되었다. (카테코리 특성)
# print(type(y)) # <class 'numpy.ndarray'>
# print(y[:25]) # 0으로 채워진 열이 생김
# print(np.unique(y[:, 0], return_counts=True)) # 모든 행의, 0번째 컬럼 (array([0.]  => 0으로만 채워진 , dtype=float32), array([581012], dtype=int64))
# print(np.unique(y[:, 1], return_counts=True)) # 모든 행의, 1번째 컬럼 (array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))
# print("======================================================")
# y = np.delete(y, 0, axis=1) # y의 0번째 axis=1(열)을 지우겠다.
# print(y.shape)  #(581012, 7) 
# print(y[:25]) # 한 번 확인, 하지만 값에 0만 나와 제대로된 판별이 힘듬
# print(np.unique(y[:, 0], return_counts=True)) # 제대로 확인해보기 위해 unique 사용


############################## 판다스 get_dummies() #####################################
# y = pd.get_dummies(y)   # numpy는 index 생성이 안되지만 판다스에서는 index 자동 생성 해준다.
# print(y[:10]) 
# print(type(y))
# y = y.values    # valuse 를 사용하면 자동 numpy로 변환,     두 가지 사용해도 됨           
# #y = y.numpy()   #애초에 y_test, y_train 의 시초가 되는 y 데이터를 아예 numpy 형식으로 변환하여 argmax 사용 시 에러나지 않게 해준다.
# print(type(y))


############################## 사이킷런 OneHotEncoder #####################################
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
#tensorflow 에서는 numpy, pandas 등 다양한 형식을 받아드림


                                                              
#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)



y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)  # y_predict은 경우 이미 텐서플로를 거쳐서 에러가 안나지만

print( 'y_predict(예측값)' , y_predict[:20])

y_test = np.argmax(y_test, axis=1)

print( 'y_test(원래값)' , y_test) # y_test는 자료형은 그대로 판다스의 형태이기 때문에, numpy 와 형에러 발생

acc = accuracy_score(y_test, y_predict) 

print(acc)



