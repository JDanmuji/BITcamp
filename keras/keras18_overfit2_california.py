import numpy as np 
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing


#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                                                                       
)


#2. 모델 구성
inputs = Input(shape=(8, ))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128, activation='relu') (hidden1)
hidden3 = Dense(64, activation='relu') (hidden2)
hidden3 = Dense(32, activation='relu') (hidden2)
hidden3 = Dense(16, activation='relu') (hidden2)
hidden4 = Dense(8) (hidden3)
hidden4 = Dense(4) (hidden3)
output = Dense(1) (hidden4)

model = Model(inputs=inputs, outputs=output)

import time

model.compile(loss='mse', optimizer='adam')                                 
start = time.time()                                                                            
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1) #fit 이 return 한다.
end = time.time()

#3. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

print('============================================')
print(hist) # <keras.callbacks.History object at 0x00000258175F20A0>
print('============================================')
print(hist.history) # loss, vel-loss 의 변화 형태(딕셔너리 형태|key-value) , value의 형태가 list
print('============================================')
print(hist.history['loss'])
print('============================================')
print(hist.history['val_loss'])
print('============================================')
print('loss : ', loss)
print('============================================')
print('걸린시간 : ', end - start)


plt.figure(figsize=(9,6))
# x 명시 안해도 됨
# hist loss 사용, 색은 red, 선모양은 ., y 선의 이름은 loss
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# hist val_loss 사용, 색은 blue, 선모양은 ., x 선의 이름은 val_loss
plt.plot(hist.history['val_loss'], c='blue',  marker='.' , label='val_loss' )
# 차트 gird 생성
plt.grid() 
# x 축 이름 
plt.xlabel('epochs')
# y 축 이름 
plt.ylabel('loss')
# 차트 제목
plt.title('california loss')
# 그래프 선 이름 표
plt.legend()
#plt.legend(loc='upper right')  그래프 선 이름 표, 위치
# 차트 창 띄우기
plt.show()




'''



'''