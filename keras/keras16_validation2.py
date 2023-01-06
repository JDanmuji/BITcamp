import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] 잘라봐!
# x_train = np.array(range(1, 11))
# y_train = np.array(range(1, 11))
# x_test = np.array([11, 12, 13])
# y_test = np.array([11, 12, 13])
# x_validation = np.array([14, 15, 16])
# y_validation = np.array([14, 15, 16])

x_train = x[:-6] 

x_test = x[-6:-3]
y_train = y[:-6]
y_test = y[-6:-3]
x_validation = x[-3:]
y_validation = y[-3:]


print(x_train)
print(x_test)
print(y_train)
print(y_test)
print(x_validation)
print(y_validation)



#2. 모델 
model = Sequential()
model.add(Dense(10, input_dim=1))
#model.add(Dense(3, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          # 데이터 검증 (훈련하고 검증하고)
          #validation_data=(x_validation, y_validation)
          )
            

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])

print('17의 예측값 : ', result)

