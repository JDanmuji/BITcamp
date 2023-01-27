import numpy as np


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping #파이썬 클래스 대문자로 시작   
from sklearn.metrics import r2_score,accuracy_score
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler

datasets = load_wine()

x = datasets.data
y = datasets.target

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    shuffle=True, # False 의 문제점은 데이터 분할 시, 한쪽으로 쏠림 현상 발생으로 데이터의 훈련도의 오차가 심해진다. 
    random_state=123, # random_state 를 사용 시, 분리된 데이터가 비율이 안맞는 현상 발생
    test_size=0.3,
    stratify=y # 분리된 데이터가 비율이 일정하게 됨, 데이터 자체(y)가 분류형 일 때만 가능 , load_boston 데이터는 회귀형 데이터라 안됨.
    
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape) #(124, 13) (54, 13)

x_train = x_train.reshape(124, 13, 1)
x_test = x_test.reshape(54, 13, 1)

print(x_train.shape, x_test.shape) #(404, 13) (102, 13)

model = Sequential()
model.add(LSTM(units=128, input_shape=(13, 1))) # 가독성
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(3, activation='softmax'))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=20, #참을성     
                              restore_best_weights=True, #최소값에 했던 지점에서 멈춤
                              verbose=1
                              )

hist = model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[earlyStopping], validation_split=0.2, verbose=1) #fit 이 return 한다.


                                 
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1) # y_predict 가장 큰 값의 자릿수 뽑음 : 예측한 값

print( 'y_predict(예측값)' , y_predict)

y_test = np.argmax(y_test, axis=1) 

print( 'y_test(원래값)' , y_test)

acc = accuracy_score(y_test, y_predict) 

print(acc)


