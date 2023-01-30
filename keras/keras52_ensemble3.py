import numpy as np 

x1_datasets = np.array([range(100),range(301, 401)]).transpose()

print(x1_datasets.shape) #(100, 2)     # 삼성전자 시가, 고가 || 데이터가 100, 2개의 컬럼

x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose()

print(x2_datasets.shape) #(100, 3)     # 아모레 시가, 고가, 종가 || 데이터가 100, 3개의 컬럼

x3_datasets = np.array([range(100,200), range(1301, 1401)]).transpose()

y1 = np.array(range(2001, 2101)) #(100, ) #삼성전자의 하루 뒤 종가
y2 = np.array(range(201, 301)) #(100, ) #아모레의 하루 뒤 종가



from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test,\
    x3_train, x3_test, y1_train, y1_test, \
    y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2 ,train_size=0.7, random_state=123
)


# y1_train = y1[:70]
# y1_test = y1[70:]
# y2_train = y2[:70]
# y2_test = y2[70:]

print(x1_train.shape, x1_test.shape)
print(x2_train.shape, x2_test.shape)
print(y1_train.shape, y1_test.shape)
print(y2_train.shape, y2_test.shape)
# y1_train, y1_test, y2_train, y2_test = train_test_split(
#    y1, y2, test_size=0.3, random_state=123
# )



#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델 1
input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11') (input1)
dense2 = Dense(12, activation='relu', name='ds12') (dense1)
dense3 = Dense(13, activation='relu', name='ds13') (dense2)
output1 = Dense(14, activation='relu', name='ds14') (dense3)


#2-2. 모델 2
input2 = Input(shape=(3,))
dense21 = Dense(11, activation='linear', name='ds21') (input2)
dense22 = Dense(12, activation='linear', name='ds22') (dense21)
output2 = Dense(13, activation='linear', name='ds23') (dense22)

#2-3. 모델 3
input3 = Input(shape=(2,))
dense31 = Dense(11, activation='relu', name='ds31') (input3)
dense32 = Dense(12, activation='relu', name='ds32') (dense31)
output3 = Dense(13, activation='linear', name='ds33') (dense32)


#2-4. 모델병합
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = Concatenate(axis=1)([output1, output2, output3])
merge2 = Dense(12, activation='relu', name='mg2') (merge1)
merge3 = Dense(13, name='mg3') (merge2)
last_output = Dense(1, name='last') (merge3)

#2-5. 모델 5 분기last_output
dense5 = Dense(11, activation='relu', name='ds51') (last_output)
dense5 = Dense(12, activation='relu', name='ds52') (dense5)
output5 = Dense(13, activation='linear', name='ds53') (dense5)

#2-6. 모델 6 분기last_output
dense6 = Dense(11, activation='relu', name='ds61') (last_output)
dense6 = Dense(12, activation='relu', name='ds62') (dense6)
output6 = Dense(13, activation='linear', name='ds63') (dense6)



model = Model(inputs=[input1, input2, input3], outputs=[output5, output6])

model.summary()

#3. 컴파일, 훈련
# print(x1_train.shape)
# print(x2_train.shape)
# print(x3_train.shape)
# print(y_train.shape)


model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=500, batch_size=8)

#4. 평가, 예측

loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])

print('loss : ', loss)

y_predict = model.predict([x1_test, x2_test, x3_test])
#result = model.predict(y_predict)
print('result : ',y_predict)


'''
loss :  [600.869140625, 9.007787704467773, 591.861328125]
'''