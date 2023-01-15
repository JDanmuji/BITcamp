from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split (
    x, y, shuffle=True, random_state=333, test_size=0.2
)

#2. 모델 구성
model =  Sequential()
#model.add(Dense(5, input_dim=13))
model.add(Dense(5, input_shape=(13, )))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

import time

model.compile(loss='mse', optimizer='adam')                                 # True : 1, False : 0, 프로그래스바 제거(진행바 사라짐) : 2, 에포(반복치)만 보여줌 : 3 ~
                                                                            # 말수가 많음 실행할 때, 코드 보여주는 게 딜레이가 생긴다.
                                                                            # 자원낭비
start = time.time()                                                                            
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2, verbose=2)
end = time.time()

#3. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss)

print('걸린시간 : ', end - start)

'''
[verbose = 0]
loss :  40.98577880859375
걸린시간 :  9.872689485549927


[verbose = 1]
loss :  45.78962326049805
걸린시간 :  11.606598138809204

[verbose = 2] 
loss :  45.07732009887695
걸린시간 :  10.344038963317871


'''