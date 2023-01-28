import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

# 1. 데이터 (ex, 삼성전자 주식 하나로 테스트 한다 가정)
dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  #(10, )

# y = ?

x = np.array([[1, 2, 3], 
              [2, 3, 4], 
              [3, 4, 5], 
              [4, 5, 6],
              [5, 6, 7],  
              [6, 7, 8], 
              [7, 8, 9]])

y = np.array([4, 5, 6, 7, 8, 9, 10])


print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(7, 3, 1)      # -> [[[1], [2], [3]],
                            #     [[2], [3], [4]],  ... ]
 
print(x.shape)  #(7, 3, 1)

# 2. 모델 구성 rnn = 2차원, rnn의 장기의존성을 해결하기 위해 LSTM이 탄생
model = Sequential()
#model.add(SimpleRNN(units=10, input_shape=(3, 1))) # (N, 3, 1) -> ([batch, timesteps, feature])
model.add(LSTM(units=10, input_shape=(3, 1))) # 가독성
model.add(Dense(32, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

# SimpleRNN
# 10 * (10 + 1 + 1) = 120
# units * (feature + bias + units) = params

# LSTM
# 4 * (10 * (10 + 1 + 1)) = 480
# 4 * (feature + bias + units) = params

model.summary()

# #3. 컴파일 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=1000, batch_size=1)

# # 4. 평가, 예측
# loss = model.evaluate(x, y)
# print('loss : ', loss)
# y_pred = np.array([8, 9, 10]).reshape(1, 3, 1)
# result = model.predict(y_pred)

# print('[8, 9, 10]의 결과 : ',  result)
