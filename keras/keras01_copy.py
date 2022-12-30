
#numpy 배열 및 벡터 계산하는 파이썬 라이브러리
import numpy as np 

#1. 데이터
#배열값 넣고
x = np.array([1, 2, 3]) 
y = np.array([1, 2, 3])

#2. 모델구성
#keras :  딥러닝 API 라이브러리
#tensorflow + keras(통합 되었음), tensorflow 안의 keras 중 model, layer 에서 Sequential, Dense 클래스를 import 함
#순차적 모델 (딥러닝에서 순차적을 모델로 만들 수 있음)
from tensorflow.keras.models import Sequential
#Dense 란? Dense 레이어, 뉴런의 입출력을 연결해줌
#뉴런이란? 인공신경망을 의미, 다수의 입력 신호를 받아 하나의 신호를 출력
#신경망 작동 원리가 일차함수, y = wx + b
#Dense 란? 인공신경망의 입출력을 연결해주며, 고로, 일차함수의 입출력을 만들어준다.
from tensorflow.keras.layers import Dense 


#모델을 Sequential로 선언
#Sequential Model : 레이어를 선형으로 연결하여 구성
model = Sequential()
#add 메서드를 사용하여 위의 x, y 값을 Dense 클래스를 통해 넣기

#      Dense(1, input_dim=1)
#      Dense(units(출력 뉴런의 수), input_dim=x(입력 뉴런의 수))


model.add(Dense(1, input_dim=1))  # y = ax + b
                #output(y), input(x)

#3. 컴파일, 훈련

# keras 모델을 컴파일하는 데 필요한 인수, loss, optimizer
# loss : 손실 함수는 값을 예측할 때, 데이터에 대한 예측값과 실제의 값을 비교하는 함수
#        모델을 훈련시킬 때 오류를 최소화 시키기 위해 사용, 회귀에서 사용
# 종류 중 하나 'mae'
# mae : mean absolute error (평균 절대 오차)
# optimizer : 손실 함수를 통해 얻은 손실값으로부터 모델을 업데이트 하는 방식
# 종류 중 하나 'adam'
# adam :  adam 알고리즘, 최적화 알고리즘
model.compile(loss='mae', optimizer='adam')  
# fit() 함수 : 케라스에서 만든 모델을 학습할 때 쓰는 것
# x : 입력 데이터
# y : 라벨 값
# epochs : 학습반복횟수
model.fit(x, y, epochs=2000) 


#4. 평가, 예측
#predict 예측하다.
result = model.predict([4]) #[4]에 대한 값을 가지고 예측
print('결과 : ', result)



