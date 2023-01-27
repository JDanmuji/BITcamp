import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path = './_data/ddarung/'                    
                                            #index 컬럼은 0번째
train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # [715 rows x 9 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)     #[1459 rows x 10 columns]
submission = pd.read_csv(path + 'submission.csv', index_col=0)  #[715 rows x 1 columns], 2개중 count 컬럼을 제외한 나머지 1개
train_csv = train_csv.interpolate(method='linear', limit_direction='forward')

x = train_csv.drop(['count'], axis=1) # 10개 중 count 컬럼을 제외한 나머지 9개만 inputing
y = train_csv['count']




x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2, shuffle=True, random_state=123
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


test_csv = scaler.transform(test_csv)

print(x_train.shape, x_test.shape) #(1167, 9) (292, 9)

x_train = x_train.reshape(1167, 9, 1)
x_test = x_test.reshape(292, 9, 1)

print(x_train.shape, x_test.shape) #(404, 13) (102, 13)

model = Sequential()
model.add(LSTM(units=128, input_shape=(9, 1))) # 가독성
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)


#3. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

#제출
y_submit = model.predict(test_csv)

# to.csv() 를 사용해서 submission_0105.csv를 완성하시오.
submission['count'] = y_submit
submission.to_csv(path + 'submission_0105.csv')




