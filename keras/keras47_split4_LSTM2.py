import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

a = np.array(range(1,101))
x_predict = np.array(range(96, 106))
# 예상 y = 100, 107

timesteps = 5   # x는 4개, y는 1개

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)
 
bbb = split_x(a, timesteps)
x = bbb[:, :-1]
y = bbb[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.2, shuffle=True, random_state=123
)


print(x_train.shape, x_test.shape) #(76, 4) (20, 4)
#x_train = x_train.reshape(96, 4, 1)    
#x_test = x_test.reshape(96, 4, 1)    

x_train = x_train.reshape(76 , 2, 2, 1)
x_test = x_test.reshape(20 , 2, 2, 1)

timesteps = 4

x_predict = split_x(x_predict, timesteps)
x_predict = x_predict.reshape(7, 2, 2, 1)

# 모델구성

model = Sequential()
model.add(Conv2D(1024, (2,1), input_shape=(2, 2, 1)))
model.add(Flatten()) 
model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))
model.add(Dense(24, activation = 'linear'))
model.add(Dense(1, activation = 'softmax'))

model.summary()
# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='loss', mode='min',patience=100,
                  restore_best_weights=True,
                   verbose=1)

filepath = './_save/MCP/'
filename = '{epoch:04d}-{loss:.4f}.hdf5'

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

mcp = ModelCheckpoint(monitor='loss', mode = 'auto', verbose = 1,
                        save_best_only=True,
                        filepath = filepath + 'k47_2_' + date +'_'+ filename)

model.fit(x_train, y_train, epochs=10000, batch_size=32,
          callbacks=[es],verbose=1)

# 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)

result = model.predict(x_test)


print('[100, 107]의 결과 : ', result )