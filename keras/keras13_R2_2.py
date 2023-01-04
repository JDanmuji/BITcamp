# 실습
# 1. R2를 음수 가 아닌 0.5 이하로 줄이기
# 2. 데이터는 건들지 말 것
# 3. 레이어는 인풋 아웃풋 포함 7개 이상
# 4. batch_size=1
# 5. 히든레이어의 노드는 각각 10개 이상 100개 이하
# 6. train 70%
# 7. epoch 100번 이상
# 8. loss 지표는 mse 또는 mae

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,21))
y = np.array(range(1,21))


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                                                                       
)

#2. 모델 구성 // 과적합, 많은 계산량, 많은 학습 데이터
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(4))
model.add(Dense(300))
model.add(Dense(1000))
model.add(Dense(4))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=1)


#4. 평가
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

y_predict = model.predict(x_test)

print("===================================")
print(y_test)
print(y_predict)
print("===================================")

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : " , RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# RMSE :  3.8315291146057255