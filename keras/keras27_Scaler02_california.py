import numpy as np 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.callbacks import EarlyStopping #파이썬 클래스 대문자로 시작   
from sklearn.preprocessing import MinMaxScaler, StandardScaler



#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                                                                       
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
inputs = Input(shape=(8, ))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128, activation='relu') (hidden1)
hidden3 = Dense(64, activation='relu') (hidden2)
hidden4 = Dense(8) (hidden3)
output = Dense(1) (hidden4)

model = Model(inputs=inputs, outputs=output)



#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start = time.time()
                                                   
                                                                                               
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=10, #참을성     
                              restore_best_weights=True, 
                              verbose=1
                              )

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[earlyStopping], validation_split=0.2, verbose=1) #fit 이 return 한다.
end = time.time()

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
print("걸린 시간 : ", end - start)
print("===================================")


'''

[Scaler 적용 전]
===================================
loss :  [0.5647933483123779, 0.5547025203704834]
R2 :  0.5728670877936994
RMSE :  0.7515272379396186
걸린 시간 :  8.49163818359375
===================================


[Scaler 적용 후]
===================================
loss :  [0.2701438367366791, 0.35069575905799866]
R2 :  0.7999664484581603
RMSE :  0.5197536095223483
걸린 시간 :  35.4945969581604
===================================

'''
