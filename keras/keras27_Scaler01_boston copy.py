import numpy as np 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터
dataset = load_boston() #numpy 값을 모여서 사이킷 런 유틸로 묶어놓은 것

x = dataset.data #보스턴 집 값
y = dataset.target



x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    test_size=0.2, shuffle=True
)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
    test_size=0.2, shuffle=True, random_state=333
)



scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

x_train = scaler.fit_transform(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)


#2. 모델구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(13,)))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(1, activation='softmax')) 



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_data=(x_validation, y_validation))

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ' , mse)
print('mae : ' , mae)


y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))


r2 = r2_score(y_test, y_predict)


print("===================================")

print("R2 : ", r2)
print("RMSE : " , RMSE(y_test, y_predict))
print("===================================")


'''

mse :  681.2109985351562
mae :  23.94444465637207
===================================
R2 :  -5.314843256456973
RMSE :  26.100018920574
===================================

mse :  252.65371704101562
mae :  15.025489807128906
===================================
R2 :  -8.39639430827756
RMSE :  15.895084947561497
===================================
'''



