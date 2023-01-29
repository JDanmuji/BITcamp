import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler


a = np.array(range(1, 11))
timesteps = 5

def split_x(dataset, timesteps) :
    
    aaa = []
                    # len : 문자열 길이 함수
    #주어진 데이터 셋의 길이의 원하는 갯수만큼 자르는 for문, 
    #range의 +1를 하는 이유 : 자르는 개수만큼 +1 를 해주면 총 길이가 나옴
    for i in range(len(dataset) - timesteps + 1) :  # 10 - 5 + 1 =6
        subset = dataset[i : (i + timesteps)] # 0 : 0 + 5
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)

# print(bbb)
# print(bbb.shape)

x = bbb[:, :-1]
y = bbb[:, -1]

# print(x, y)

print(x.shape, y.shape) #(6, 4) (6,)

x = x.reshape(6, 4, 1)     

# 실습
# LSTM 모델 구성
#x_predict = np.array([7, 8, 9, 10])

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
print(x.shape, y.shape)
model.fit(x, y, epochs=500, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = np.array([7, 8, 9, 10]).reshape(1, 4, 1)
result = model.predict(y_pred)

print('[7, 8, 9, 10]의 결과 : ',  result)




