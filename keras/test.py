import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler

a = np.array(range(1, 101))

x_predict = np.array(range(96, 106))

timesteps = 5

def split_x(dataset, timesteps) :
    
    aaa = []
    
    for i in range(len(dataset) - timesteps + 1) :  # 10 - 5 + 1 =6
        subset = dataset[i : (i + timesteps)] # 0 : 0 + 5
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)

x = bbb[:, :-1]
y = bbb[:, -1]

print(x.shape, y.shape) #(96, 4) (96,)


x = x.reshape(96, 4, 1)    

timesteps=4


x_predict = split_x(x_predict, timesteps)

print(x_predict.shape) # (7, 4)

x_predict = x_predict.reshape(7, 4, 1)

model = Sequential()
model.add(LSTM(units=128, input_shape=(4, 1))) # 가독성
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=6)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

x_pred = x_predict.reshape(7, 4, 1)
result = model.predict(x_pred)

print( np.array(range(96, 106)), '의 결과 : ',  result)


