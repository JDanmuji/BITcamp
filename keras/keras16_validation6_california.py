# 실습 
# R2 0.55 ~ 0.6 이상
import numpy as np 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                                                                       
)

#2. 모델구성
inputs = Input(shape=(8, ))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128, activation='relu') (hidden1)
hidden3 = Dense(64, activation='relu') (hidden2)
hidden3 = Dense(32, activation='relu') (hidden2)
hidden3 = Dense(16, activation='relu') (hidden2)
hidden4 = Dense(8) (hidden3)
hidden4 = Dense(4) (hidden3)
output = Dense(1) (hidden4)

model = Model(inputs=inputs, outputs=output)



#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.25)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)


y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))


r2 = r2_score(y_test, y_predict)


print("===================================")
print('loss : ' , loss)
print("R2 : ", r2)
print("RMSE : " , RMSE(y_test, y_predict))
print("걸린 시간 : ", end - start)
print("===================================")


'''

[batch_size=32]

===================================
loss :  [0.4915376305580139, 0.5304100513458252]
R2 :  0.6282677411627802
RMSE :  0.7010973612070512
===================================



[cpu]
loss :  [0.47160282731056213, 0.5052147507667542]
R2 :  0.643343658506472
RMSE :  0.686733419079916
걸린 시간 :  42.19237685203552

[gpu]
loss :  [0.47861427068710327, 0.5232479572296143]
R2 :  0.6380411738736391
RMSE :  0.6918194897918106
걸린 시간 :  145.63760232925415

'''
