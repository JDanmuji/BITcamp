import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 5, 4])

# 2. 모델구성, 심층신경망(딥러닝)
# 노드의 깊이가 많을수록 성능은 좋아지지만, 여러 요소를 신경써야됨
# 하이퍼 파라미터 튜닝
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200)

# 4. 평가, 예측
result = model.predict([6])
print('6의 결과 : ', result)


