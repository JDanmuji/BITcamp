import numpy as np 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터
dataset = load_boston() 

x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2, shuffle=True, random_state=333
)


scaler = MinMaxScaler()
#scaler.fit(x_train) 
#x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)


# #2. 모델구성(순차형)
model = Sequential()
model.add(Dense(50, input_dim=13, activation = 'linear'))
model.add(Dense(40, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(20, activation = 'linear'))
model.add(Dense(1, activation = 'linear'))
model.summary() #Total params: 4,611

#2. 모델구성(함수형)
# input1 = Input(shape=(13, ))
# dense1 = Dense(50, activation='relu') (input1)
# dense2 = Dense(40, activation='sigmoid') (dense1)
# dense3 = Dense(30, activation='relu') (dense2)
# dense4 = Dense(20, activation='linear') (dense3)
# output1 = Dense(1, activation='linear') (dense4)

# model = Model(inputs=input1, outputs=output1)
# model.summary() #Total params: 4,611

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping #파이썬 클래스 대문자로 시작   

#earlyStopping 약점 : 5번을 참고 끊으면 그 순간에 weight가 저장 (끊는 순간)
                                                    
                                                                                               
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=10, #참을성     
                              restore_best_weights=True, 
                              verbose=1
                              )
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2)

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ' , mse)
print('mae : ' , mae)


y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))


r2 = r2_score(y_test, y_predict)


print("===================================")

print("R2 : ", r2)
print("RMSE : " , RMSE(y_test, y_predict))
print("===================================")


'''


'''




