from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split



# rmse 함수를 만들어서 아래 코드에 삽입

#1. 데이터
x = np.array(range(1,21))
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                                                                       
)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)



y_predict = model.predict(x_test)

print("===================================")
print(y_test)
print(y_predict)
print("===================================")

'''
    [mae의 결과]
    loss :  3.146573305130005
    
    [mse의 결과]
    loss :  15.29187297821045
    
'''

from sklearn.metrics import mean_squared_error


def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : " , RMSE(y_test, y_predict))



# RMSE :  3.95773894675322
# R2 :  0.6282840391936042