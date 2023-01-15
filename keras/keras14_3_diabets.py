#[과제, 실습]
# R2 0.62 이상
import numpy as np 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes

# 1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target


print(x)
print(x.shape) # (442, 10)
print(y)
print(y.shape) # (442,)



x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.9, shuffle=True, random_state=123                                                                                                       
)

#2. 모델구성
inputs = Input(shape=(10, ))
hidden1 = Dense(256) (inputs)
hidden2 = Dense(128) (hidden1)
hidden3 = Dense(64) (hidden2)
hidden4 = Dense(64) (hidden3)
hidden5 = Dense(10) (hidden4)
hidden6 = Dense(5) (hidden5)
output = Dense(1) (hidden6)

model = Model(inputs=inputs, outputs=output)

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=700, batch_size=4, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)


y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))


r2 = r2_score(y_test, y_predict)


print("===================================")
print('loss : ' , loss)
print("R2 : ", r2)
print("RMSE : " , RMSE(y_test, y_predict))
print("===================================")


'''
[train_size=0.9]
===================================
loss :  [2576.796142578125, 38.35049819946289]
R2 :  0.6136092423759105
RMSE :  50.76215388503177
===================================




'''