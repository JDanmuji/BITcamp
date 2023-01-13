import numpy as np 
import datetime

from tensorflow.keras.callbacks import EarlyStopping ,ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential, Model

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

path = './_save/'

# 1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2, shuffle=True, random_state=123
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)


#2. 모델구성
inputs = Input(shape=(10, ))
hidden1 = Dense(256, activation='relu') (inputs)
drop1 = Dropout(0.5) (hidden1)
hidden2 = Dense(128, activation='relu') (drop1)
drop2 = Dropout(0.4) (hidden2)
hidden3 = Dense(64, activation='relu') (drop2)
drop3 = Dropout(0.3) (hidden3)
hidden4 = Dense(32, activation='relu') (drop3)
drop4 = Dropout(0.25) (hidden4)
hidden5 = Dense(16, activation='relu') (drop4)
drop5 = Dropout(0.15) (hidden5)
hidden6 = Dense(8) (drop5)
drop6 = Dropout(0.1) (hidden6)
hidden7 = Dense(4) (drop6)
drop7 = Dropout(0.05) (hidden7)
output = Dense(1) (drop7)

model = Model(inputs=inputs, outputs=output)


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])
                                                   
                                                                                               
es = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=100, #참을성     
                              restore_best_weights=True, 
                              verbose=1
                              )


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0037-0.0048.hdf

mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = filepath + 'k31_03_' + date + '_'+ filename)

model.fit(x_train, 
          y_train, 
          epochs=1000, 
          batch_size=8, 
          validation_split=0.25, 
          callbacks=[es, mcp])

model.save(path + 'keras31_dropout03_save_model.h5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)


print("===================================")
print('loss : ' , loss)
print("R2 : ", r2)
print("===================================")


'''
===================================
loss :  [44.49851989746094, 2828.712890625]
R2 :  0.5510098875977331
===================================


'''