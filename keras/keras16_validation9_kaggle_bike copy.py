#https://www.kaggle.com/competitions/bike-sharing-demand

#RMES

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns #%matplotlib inline


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
        
#1. 데이터
path = './_data/bike/'

train_df = pd.read_csv(path + 'train.csv', index_col=0)
test_df = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

#train_csv = train_csv.interpolate(method='linear', limit_direction='forward')






train_df['datetime'] = pd.to_datetime(train_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
train_df['year'] = train_df['datetime'].apply(lambda x: x.year)
train_df['month'] = train_df['datetime'].apply(lambda x: x.month)
train_df['day'] = train_df['datetime'].apply(lambda x: x.day)
train_df['hour'] = train_df['datetime'].apply(lambda x: x.hour)

test_df['year'] = test_df['datetime'].apply(lambda x: x.year)
test_df['month'] = test_df['datetime'].apply(lambda x: x.month)
test_df['day'] = test_df['datetime'].apply(lambda x: x.day)
test_df['hour'] = test_df['datetime'].apply(lambda x: x.hour)

x = train_df.drop(['count', 'casual', 'registered'], axis=1)
y = train_df['count']


x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    train_size=0.85, test_size=0.15, shuffle=False
)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
   test_size=0.2, shuffle=False
)

#2. 모델 구성
# activation='liner' (Default)
inputs = SimpleRNN(shape=(8,))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128) (hidden1)
hidden3 = Dense(64) (hidden2)
hidden4 = Dense(32, activation='relu') (hidden3)
hidden5 = Dense(16) (hidden4)
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
R2 :  0.2462426148793062
RMSE :  156.48429147670618
'''


