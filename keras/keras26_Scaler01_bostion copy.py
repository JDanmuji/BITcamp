import numpy as np 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler


# 1. 데이터
dataset = load_boston() #numpy 값을 모여서 사이킷 런 유틸로 묶어놓은 것

x = dataset.data #보스턴 집 값
y = dataset.target


scaler = MinMaxScaler()
scaler.fit(x) 
x = scaler.transform(x) # 실질적으로 값을 변환, 값이 바뀜

# print('최소값 : ' , np.min(x)) # 최소값 :  0.0
# print('최대값 : ' , np.max(x)) # 최대값 :  1.0


x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    test_size=0.2, shuffle=False
)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
    test_size=0.2
)



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

[변환전]

[변환후]
mse :  495.10791015625
mae :  20.569766998291016
===================================
R2 :  -5.877207980782226
RMSE :  22.251020358103673
===================================


mse :  553.7828979492188
mae :  21.731393814086914
===================================
R2 :  -5.792435086570172
RMSE :  23.53259244062889
===================================

'''



