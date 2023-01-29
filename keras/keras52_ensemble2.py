import numpy as np 
import pandas as pd


# 1. 데이터
path = './_data/ensemble/'                    
                              
                           
                              
                                            #index 컬럼은 0번째
x1_datasets = pd.read_csv(path + 'samsung.csv', encoding='cp949')   # [715 rows x 9 columns]
x2_datasets = pd.read_csv(path + 'amore.csv', encoding='cp949')     #[1459 rows x 10 columns]



# 결측치 처리 
# 1. 선형 방법을 이용하여 결측치
# train_csv = train_csv.interpolate(method='linear', limit_direction='forward')

# x = train_csv.drop(['count'], axis=1) # 10개 중 count 컬럼을 제외한 나머지 9개만 inputing
# y = train_csv['count']


x1_datasets = x1_datasets[['일자', '시가', '고가', '저가', '종가']]
x2_datasets = x2_datasets[['일자', '시가', '고가', '저가', '종가']]

y = x1_datasets[['시가']]
x2_datasets = x2_datasets[0 : 1980]


print(x1_datasets.shape) #(7, 1980)
     
print(x2_datasets.shape) #(7, 1980)


y = y[0:1980]

print(y) #(2220, 7)

x1_datasets = x1_datasets.values
x2_datasets = x2_datasets.values
y = y.values

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.7, random_state=123
)

print(x1_train.shape, x2_train.shape, y_train.shape)  #(70, 2) (70, 3) (70,)
print(x1_test.shape, x2_test.shape, y_test.shape) #(30, 2) (30, 3) (30,)


#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델 1
input1 = Input(shape=(5,))
dense1 = Dense(11, activation='relu', name='ds11') (input1)
dense2 = Dense(12, activation='relu', name='ds12') (dense1)
dense3 = Dense(13, activation='relu', name='ds13') (dense2)
output1 = Dense(14, activation='relu', name='ds14') (dense3)


#2-2. 모델 2
input2 = Input(shape=(5,))
dense21 = Dense(11, activation='linear', name='ds21') (input2)
dense22 = Dense(12, activation='linear', name='ds22') (dense21)
output2 = Dense(13, activation='linear', name='ds23') (dense22)

#2-3. 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(12, activation='relu', name='mg2') (merge1)
merge3 = Dense(13, name='mg3') (merge2)
last_output = Dense(1, name='last') (merge3)


model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')


model.fit([x1_train, x2_train], y_train, epochs=500, batch_size=8)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)

print('loss : ',loss)