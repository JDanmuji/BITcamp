import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. 데이터
#아래와 같이 잘라서 넣어줄 수 있지만, 시간과 비용이 많이 든다.
#x = np.array([1], [2], [3], [4], [5], [6])

x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 5, 4, 6])

# 2. 모델구성, 심층신경망(딥러닝)
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(13))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#batch 사이즈는 fit에서 관리
model.fit(x, y, epochs=10, batch_size=1)

# 4. 평가, 예측
#loss 가 기준이다. (predict으로 평가하면 안된다.)
loss, acc = model.evaluate(x, y, batch_size=1) #훈련된 데이터가 들어가면 안된다. 
result = model.predict([6])
print('loss : ', loss)
print('acc : ', acc)
print('6의 결과 : ', result)


