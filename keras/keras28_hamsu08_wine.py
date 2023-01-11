import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler


import pandas as pd

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    shuffle=True, # False 의 문제점은 데이터 분할 시, 한쪽으로 쏠림 현상 발생으로 데이터의 훈련도의 오차가 심해진다. 
    random_state=333, # random_state 를 사용 시, 분리된 데이터가 비율이 안맞는 현상 발생
    test_size=0.3,
    stratify=y # 분리된 데이터가 비율이 일정하게 됨, 데이터 자체(y)가 분류형 일 때만 가능 , load_boston 데이터는 회귀형 데이터라 안됨.
    
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)


#2. 모델구성 # 분류형 모델
inputs = Input(shape=(13, ))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128, activation='sigmoid') (hidden1)
hidden3 = Dense(64, activation='relu') (hidden2)
hidden4 = Dense(32, activation='linear') (hidden3)
hidden5 = Dense(16, activation='linear') (hidden4)
output = Dense(3, activation='softmax') (hidden5)

model = Model(inputs=inputs, outputs=output)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                                                                                             
model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.25, verbose=1)

                                                              
#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)



y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1) # y_predict 가장 큰 값의 자릿수 뽑음 : 예측한 값

print( 'y_predict(예측값)' , y_predict)

y_test = np.argmax(y_test, axis=1) # y_test : y_test 값, (원 핫 인코딩을 진행했기 때문에, 다시 원복) 

print( 'y_test(원래값)' , y_test)

acc = accuracy_score(y_test, y_predict) 

print(acc)

'''

[Scaler 변경 전]
loss :  0.13796885311603546
accuracy :  0.9814814925193787


[Scaler 변경 후]
loss :  0.03192202374339104
accuracy :  1.0


'''