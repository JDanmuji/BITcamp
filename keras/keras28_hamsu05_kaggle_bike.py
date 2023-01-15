#https://www.kaggle.com/competitions/bike-sharing-demand

#RMES

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path = './_data/bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

#train_csv = train_csv.interpolate(method='linear', limit_direction='forward')
x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']



x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    train_size=0.85, test_size=0.15, shuffle=True
)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
   test_size=0.2, shuffle=True, random_state=123
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

test_csv = scaler.transform(test_csv)

#2. 모델 구성
# activation='liner' (Default)
inputs = Input(shape=(8,))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128, activation='relu') (hidden1)
hidden3 = Dense(64, activation='relu') (hidden2)
hidden4 = Dense(32, activation='relu') (hidden3)
hidden5 = Dense(16, activation='relu') (hidden4)
hidden6 = Dense(8) (hidden5)
output = Dense(1) (hidden6)

model = Model(inputs=inputs, outputs=output)


#3. 컴파일, 훈련
model.compile(loss="mae", optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32,  validation_data=(x_validation, y_validation))

#4. 예측, 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

#제출
y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + 'submission_0106.csv')

print("===================================")
print(y_test)
print(y_predict)
print("submit : ", y_submit) 
print("R2 : " , r2)
print("RMSE : ", RMSE(y_test, y_predict))
print("===================================")

'''
[Scaler 적용 전]
R2 :  0.28876003210238255
RMSE :  148.34438609945983
===================================


[Scaler 적용 후 - MinMaxScaler]
R2 :  0.31660096718979125
RMSE :  149.2092754089569
===================================
R2 :  0.272831456086538
RMSE :  153.2706070224963
===================================

[Scaler 적용 후 - StandardScaler]
R2 :  0.27870922945430987
RMSE :  154.2892045456745
===================================
R2 :  0.30646404413019157
RMSE :  154.75422028698256


'''


