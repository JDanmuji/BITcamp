import numpy as np 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from tensorflow.keras.callbacks import EarlyStopping #파이썬 클래스 대문자로 시작   
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target


x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    train_size=0.85, test_size=0.15, shuffle=False
)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
    test_size=0.2, shuffle=False
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)


#2. 모델구성
inputs = Input(shape=(10, ))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128, activation='relu') (hidden1)
hidden3 = Dense(64, activation='relu') (hidden2)
hidden4 = Dense(32, activation='relu') (hidden3)
hidden5 = Dense(16) (hidden4)
hidden6 = Dense(8) (hidden5)
output = Dense(1) (hidden6)

model = Model(inputs=inputs, outputs=output)

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])
                                                   
                                                                                               
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=10, #참을성     
                              restore_best_weights=True, 
                              verbose=1
                              )

hist = model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[earlyStopping], validation_split=0.2, verbose=1) #fit 이 return 한다.
model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_validation, y_validation))

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
[Scaler 적용 전]
===================================
loss :  [46.39445877075195, 3832.401611328125]
R2 :  0.2656540183646532
RMSE :  61.90639446574987
===================================


[Scaler 적용 후]
===================================
loss :  [47.520347595214844, 3566.592529296875]
R2 :  0.3165870755588649
RMSE :  59.72095490008722
===================================



'''