import numpy as np

from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping #파이썬 클래스 대문자로 시작   
from sklearn.metrics import r2_score,accuracy_score

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2, shuffle=True, random_state=123
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape, x_test.shape) #(455, 30) (114, 30)

x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)

print(x_train.shape, x_test.shape) #(404, 13) (102, 13)

model = Sequential()
model.add(LSTM(units=128, input_shape=(30, 1))) # 가독성
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=20, #참을성     
                              restore_best_weights=True, #최소값에 했던 지점에서 멈춤
                              verbose=1
                              )

hist = model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[earlyStopping], validation_split=0.2, verbose=1) #fit 이 return 한다.


#3. 평가, 예측
#loss = model.evaluate(x_test, y_test)
loss, accuracy = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
y = y_predict[:10] # 정수형으로 바꿔줘야겠죵?
print(y_test[:10])
#accuracy_score 보면 y_predict(실수), y_test(이진코드) 자료형이 안맞음

y = list(map(int, y))
print(y)

# 자료형 변환
y_predict = list(map(int, y_predict))
y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)

print(y_predict)
print(list(map(int, y_predict[:10])))


print('============================================')
print(hist) # <keras.callbacks.History object at 0x00000258175F20A0>
print('============================================')
print(hist.history) # loss, vel-loss 의 변화 형태(딕셔너리 형태|key-value) , value의 형태가 list
print('============================================')
print(hist.history['loss'])
print('============================================')
print(hist.history['val_loss'])
print('============================================')
print('loss : ', loss, ' accuracy : ', accuracy )
print('============================================')
print(' accuracy_score : ', acc )
print('============================================')

s


