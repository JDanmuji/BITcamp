# [실습]
# 1. train 0.7 이상
# 2. R2 : 0.8 이상/RMSE 사용
# 평가지표 : R2, RMSE
import numpy as np 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 1. 데이터
dataset = load_boston()
x = dataset.data #보스턴 집 값
y = dataset.target


x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    train_size=0.85, test_size=0.15, shuffle=False
)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
    test_size=0.2, shuffle=False
)



#2. 모델구성
inputs = Input(shape=(13, )) # x 가 0 이하일 때를 차단하여 아무 값도 출력하지 않고 0 을 출력
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128, activation='relu') (hidden1)
hidden3 = Dense(64, activation='relu') (hidden2)
hidden4 = Dense(32) (hidden3)
hidden5 = Dense(8) (hidden4)
hidden6 = Dense(4) (hidden5)
output = Dense(1) (hidden6)

model = Model(inputs=inputs, outputs=output)



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=32,  validation_data=(x_validation, y_validation))

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
loss :  [25.3370304107666, 3.836290121078491]
R2 :  0.6865318459553265
RMSE :  5.033590437402777


[batch_size=4]
loss :  [18.5551815032959, 3.1334645748138428]
R2 :  0.7704364575140819
RMSE :  4.307572756982908

'''



