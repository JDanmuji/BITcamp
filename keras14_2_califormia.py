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

print(x)
print(x.shape) # (20640, 8)
print(y)
print(y.shape) # (20640, 8)



x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                                                                       
)

#2. 모델구성
inputs = Input(shape=(8, ))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128) (hidden1)
hidden3 = Dense(64) (hidden2)
hidden4 = Dense(10) (hidden3)
output = Dense(1) (hidden4)

model = Model(inputs=inputs, outputs=output)



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=32)

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
print("===================================")


'''

[batch_size=32]

===================================
loss :  [0.4915376305580139, 0.5304100513458252]
R2 :  0.6282677411627802
RMSE :  0.7010973612070512
===================================


'''
