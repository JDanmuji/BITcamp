import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# x, y의 값을 넣고 output 구하기
# 여러가지 기초데이터를 주고 output 예상하기
# ex) 주식 시세를 맞추기 위해, 신생아 출생률, 날씨 등의 데이터를 넣고 
# 예측하기 (주식 시세를 맞출 수 없지만 예측은 가능)

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)]) # 0 ~ (x-1), 0~9
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]])


print(x.shape)
print(y.shape)

x = x.T
y = y.T

model = Sequential()
model.add(Dense(40, input_dim=3))
model.add(Dense(30))
model.add(Dense(35))
model.add(Dense(35))
model.add(Dense(35))
model.add(Dense(4000))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(2))

model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100)

loss = model.evaluate(x, y)
print('loss', loss)

result = model.predict([[10, 31, 211]])
print('[10, 31, 211]의 예측값 : ', result)









