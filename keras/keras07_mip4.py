import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([range(10)]) # (10,) (10, 1)
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])

print(x.shape)
print(y.shape)

x = x.T
y = y.T

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(80))
model.add(Dense(10000))
model.add(Dense(5000))
model.add(Dense(400))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(8))
model.add(Dense(3))

model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100)

loss = model.evaluate(x, y)
print('loss : ' , loss)

result = model.predict([10])
print('[10]의 예측값 : ', result)

result = model.predict([1.4])
print('[1.4]의 예측값 : ', result)

result = model.predict([0])
print('[0]의 예측값 : ', result)



'''
loss :  0.37171703577041626
[10]의 예측값 :  [[ 1.1845979e+01  2.0443289e+00 -8.0151483e-03]]
[1.4]의 예측값 :  [[2.5810394 1.0928242 7.070495 ]]
[0]의 예측값 :  [[1.0727946  0.93792856 8.222812  ]]
'''








