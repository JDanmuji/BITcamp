import numpy as np 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint     
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


path = './_save/'


# 1. 데이터
dataset = load_boston() 

x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2, shuffle=True, random_state=333
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성(함수형)
input1 = Input(shape=(13, ))
dense1 = Dense(50, activation='relu') (input1)
dense2 = Dense(40, activation='sigmoid') (dense1)
dense3 = Dense(30, activation='relu') (dense2)
dense4 = Dense(20, activation='linear') (dense3)
output1 = Dense(1, activation='linear') (dense4)

model = Model(inputs=input1, outputs=output1)
model.summary() #Total params: 4,611


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                                  
# 모델을 더 이상 학습을 못할 경우(loss, metric등의 개선이 없을 경우), 학습 도중 미리 학습을 종료시키는 콜백함수                                                                                            
es = EarlyStopping(monitor = 'val_loss', 
                   mode = 'min', 
                   patience = 100, #참을성     
                   restore_best_weights = True, 
                   verbose = 1)
# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = path + 'MCP/keras30_ModelCheckPoint1.hdf5')

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es, mcp])

model.save(path + 'keras29_1_save_model.h5')  #모델 저장 (가중치 포함 안됨)


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
[MCP 저장]
R2 :  0.8280174179284039


'''