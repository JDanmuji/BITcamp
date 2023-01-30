import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint     
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

import numpy as np
import pandas as pd

path = './_data/ensemble/'  

df1 = pd.read_csv(path + 'samsung.csv', index_col=0, 
                  header=0, encoding='cp949', sep=',', nrows=50)   #[1980 rows x 17 columns]

df2 = pd.read_csv(path + 'amore.csv', index_col=0, 
                  header=0, encoding='cp949', sep=',', nrows=50)     #[2220 rows x 17 columns]


# df1 = df1.apply(lambda x : x.str.replace(pat=r'[/]', repl=r'-', regex=True), axis=1)
# df2 = df2.apply(lambda x : x.str.replace(pat=r'[/]', repl=r'-', regex=True), axis=1)
df1 =  df1.dropna()
df2 = df2.dropna()


df1 = df1[[ '시가', '고가', '저가', '종가', '거래량' ]]
df2 = df2[[ '시가', '고가', '저가', '종가', '거래량']]

df1 = df1.dropna()
df2 = df2.dropna()
df1 = df1.apply(lambda x : x.str.replace(pat=r'[,]', repl=r'', regex=True), axis=1)
df2 = df2.apply(lambda x : x.str.replace(pat=r'[,]', repl=r'', regex=True), axis=1)
df1['거래량'] = pd.to_numeric(df1['거래량'], downcast='integer')
df2['거래량'] = pd.to_numeric(df2['거래량'], downcast='integer')

# df1[~df1.거래량.str.isdigit()]
# df2[~df2.거래량.str.isdigit()]

df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])
print(df1)
print(df2)

df1 = df1.values
df2 = df2.values
print(type(df1), type(df2))
print(df1.shape, df2.shape)

np.save('./_data/ensemble/samsung_data.npy', arr=df1)
np.save('./_data/ensemble/amore_data.npy', arr=df2)


# 1. 데이터

timesteps = 5  # x는 4개, y는 1개

# def split_x(dataset, timesteps):
#     aaa = []
#     for i in range(len(dataset) - timesteps + 1):
#         subset = dataset[i : (i + timesteps)]
#         aaa.append(subset)
#     return np.array(aaa)
 
def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column # 수정

        if y_end_number > len(dataset):  # 수정
            break
        tmp_x = dataset[i:x_end_number, :]  # 수정
        tmp_y = dataset[x_end_number:y_end_number, 3]    # 수정
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

                                                         
                              
                                            #index 컬럼은 0번째
x1_datasets = np.load('./_data/ensemble/samsung_data.npy', allow_pickle=True )
x2_datasets = np.load('./_data/ensemble/amore_data.npy',allow_pickle=True )


# x1_datasets = x1_datasets[[ '일자' ,'시가', '고가', '저가', '종가','거래량']]
# x2_datasets = x2_datasets[[ '일자' ,'시가', '고가', '저가', '종가','거래량']]

# x1_datasets = x1_datasets.apply(lambda x : x.str.replace(pat=r'[,-/]', repl=r'', regex=True), axis=1)
# x2_datasets = x2_datasets.apply(lambda x : x.str.replace(pat=r'[,-/]', repl=r'', regex=True), axis=1)

# y = x1_datasets[['시가']]
# y = y.apply(lambda x : x.str.replace(pat=r'[,-/]', repl=r'', regex=True), axis=1)
# x2_datasets = x2_datasets[0 : 1980]



# x1_datasets =x1_datasets.values
print(x1_datasets)
print(x2_datasets)


x1, y1 = split_xy5(x1_datasets, timesteps, 1)
x2, y2 = split_xy5(x2_datasets, timesteps, 1)


# x1_datasets = split_x(x1_datasets, timesteps)
# x1 = x1_datasets[:, :-1]
# y1 = x1_datasets[:, -1]



# x2_datasets = split_x(x2_datasets, timesteps)
# x2 = x2_datasets[:, :-1]
# y2 = x2_datasets[:, -1]


from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1,  test_size = 0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2,  test_size = 0.3)


print(x1_train.shape, x2_train.shape, y1_train.shape, y2_train.shape)  #(1381, 6, 5) (1381, 6, 5) (1381, 5) (1381, 5)
print(x1_test.shape, x2_test.shape, y1_test.shape, y2_train.shape) #(593, 6, 5) (593, 6, 5) (593, 5) (1381, 5)
# (1381, 7, 5) (1549, 7, 5) (1381, 1) (1549, 1)
# (592, 7, 5) (664, 7, 5) (592, 1) (1549, 1)

x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))
print(x1_train.shape) # (1381, 30)
print(x1_test.shape) # (593, 30)
print(x2_train.shape) # (1381, 30)
print(x2_test.shape) # (593, 30)

# (1383, 4, 5) (1383, 4, 5) (1383, 5) (1383, 5)
# (593, 4, 5) (593, 4, 5) (593, 5) (1383, 5)


# scaler = StandardScaler()
# x1_train = scaler.fit_transform(x1_train)
# x2_train = scaler.fit_transform(x2_train)

# x1_test = scaler.transform(x1_test)
# x2_test = scaler.transform(x2_test)


# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder() # 사이킷런 OneHotEncoder를 사용하면 toarray를 해줘야한다. (error :  Use X.toarray() to convert to a dense numpy array.)
# #ohe.fit(y)  # 가중치 사용, 훈련할 때 영향을 끼침
# #y = ohe.transform(y)            # (0, 4)        1.0
#                                 # (1, 4)        1.0
                                
# x1_train = ohe.fit_transform(x1_train) 
# x1_train = x1_train.toarray()
# x1_test = ohe.fit_transform(x1_test) 
# x1_test = x1_test.toarray()

     
# x2_train = ohe.fit_transform(x2_train) 
# x2_train = x2_train.toarray()
# x2_test = ohe.fit_transform(x2_test) 
# x2_test = x2_test.toarray()



from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x1_train)
x1_train_scaled = scaler1.transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)
scaler2 = StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)
print(x1_train_scaled[0, :])
print(x1_test_scaled[0, :])
print(x2_train_scaled[0, :])
print(x2_test_scaled[0, :])


x1_train_scaled = np.reshape(x1_train_scaled,
    (x1_train_scaled.shape[0], 5, 5))
x1_test_scaled = np.reshape(x1_test_scaled,
    (x1_test_scaled.shape[0], 5, 5))
x2_train_scaled = np.reshape(x2_train_scaled,
    (x2_train_scaled.shape[0], 5, 5))
x2_test_scaled = np.reshape(x2_test_scaled,
    (x2_test_scaled.shape[0], 5, 5))



# x1_train_scaled = np.reshape(x1_train,
#     (x1_train.shape[0], 5, 5))
# x1_test_scaled = np.reshape(x1_test,
#     (x1_test.shape[0], 5, 5))
# x2_train_scaled = np.reshape(x2_train,
#     (x2_train.shape[0], 5, 5))
# x2_test_scaled = np.reshape(x2_test,
#     (x2_test.shape[0], 5, 5))


print(y1_train.shape)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder() 



print(x2_train_scaled.shape)
print(x2_test_scaled.shape)

print(y1_train.shape)
# x1_train_scaled = x1_train_scaled.astype(np.float32)

# x2_train_scaled = x2_train_scaled.astype(np.float32)                                         
# y1_train = y1_train.astype(np.float32)                                         

#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델 1
input1 = Input(shape=(5,5))
dense1 =  LSTM(64)(input1)
dense2 = Dense(64, activation='relu', name='ds12') (dense1)
dense3 = Dense(64, activation='relu', name='ds13') (dense2)
output1 = Dense(32, activation='relu', name='ds14') (dense3)


#2-2. 모델 2
input2 = Input(shape=(5,5))
dense21 = LSTM(64)(input2)
dense22 = Dense(64, activation='linear', name='ds22') (dense21)
output2 = Dense(32, activation='linear', name='ds23') (dense22)

#2-3. 모델병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(24, activation='relu', name='mg2') (merge1)
merge3 = Dense(24, activation='relu', name='mg3') (merge2)
merge4 = Dense(16, name='mg4') (merge3)
last_output = Dense(1, name='last') (merge4)


model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=500, #참을성     
                              restore_best_weights=True, 
                              verbose=1
                              )
print(type(x1_test_scaled.astype('float32')))
print(x1_train_scaled.shape, type(x1_train_scaled))
print(x2_train_scaled.shape, type(x2_train_scaled))

print(y1_train.shape, type(y1_train))
print(y1_train.shape)
y1_train=np.asarray(y1_train).astype(np.float32)
y1_test=np.asarray(y1_test).astype(np.float32)
y2_test=np.asarray(y2_test).astype(np.float32)

# 모델을 저장할 때 사용되는 콜백함수
mcp = ModelCheckpoint(monitor = 'val_loss',
                      mode = 'auto',
                      verbose = 1,
                      save_best_only = True, #저장 포인트
                      filepath = path + 'MCP/keras52_ensemble2.hdf5')



model.save(path + 'keras52_ensemble2_save_model.h5')  #모델 저장 



model.fit([x1_train_scaled.astype(np.float32), x2_train_scaled.astype(np.float32)], y1_train, validation_split=0.2, 
          verbose=1, batch_size=32, epochs=1000, callbacks=[es, mcp])

#4. 평가, 예측
print(x1_test_scaled.shape)
print(x2_test_scaled.shape)
print(y1_test.shape)

loss, mse = model.evaluate([x1_test_scaled.astype(np.float32), x2_test_scaled.astype(np.float32)], y1_test, batch_size=1)

y1_predict = model.predict([x1_test_scaled.astype(np.float32), x2_test_scaled.astype(np.float32)])



print('loss : ',loss)
print('mse : ',mse)


for i in range(1):
    print('시가 : ', y1_test[i] ,'/ 예측가 : ', y1_predict[i])


