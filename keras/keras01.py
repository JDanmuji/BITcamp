import tensorflow as tf

print(tf.__version__)

import numpy as np 

#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
from tensorflow.keras.models import Sequential #순차적 모델 (딥러닝에서 순차적을 모델로 만들 수 있음)
from tensorflow.keras.layers import Dense #일차함수 y 구성 모델, layer(단계)

model = Sequential()
model.add(Dense(1, input_dim=1))  # y = ax + b
                #output(y), input(x)

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')  #loss가 mae (error) 를 사용한다, loss를 최적화 시켜주는 건 adam
model.fit(x, y, epochs=2000) #epochs : 훈련을 몇 번 시킬건지


#4. 평가, 예측
result = model.predict([4]) #[4]에 대한 값을 가지고 예측
print('결과 : ', result)



