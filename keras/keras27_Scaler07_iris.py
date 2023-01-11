from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


import pandas as pd
import tensorflow as tf

#1. 데이터
datasets = load_iris()

#print(datasets.DESCR)   #판다스 .describe() / .info()  (데이터 확인용 메서드)
#print(datasets.feature_names) # 판다스 .columns

x = datasets.data
y = datasets['target']


# One-hot Encoding 방법
# 1. keras 메서드 활용
y = to_categorical(y)

# 2. pandas의 get dummies 함수 활용
# y = pd.get_dummies(y)

# 3. tensorflow 활용
                    #라벨 개수 사용
#y = tf.one_hot(y, depth=4, on_value=Ture, off_value=False)


# 4. one_hot 벡터 return 함수 사용 (y, 빈도수)
# def one_hot_encoding(word, word_to_index):
#   one_hot_vector = [0]*(len(word_to_index))
#   index = word_to_index[word]
#   one_hot_vector[index] = 1
#   return one_hot_vector


# print(x.shape, y.shape) # (150, 4), (150, )

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    shuffle=True, # False 의 문제점은 데이터 분할 시, 한쪽으로 쏠림 현상 발생으로 데이터의 훈련도의 오차가 심해진다. 
    random_state=333, # random_state 를 사용 시, 분리된 데이터가 비율이 안맞는 현상 발생
    test_size=0.3,
    stratify=y # 분리된 데이터가 비율이 일정하게 됨, 데이터 자체(y)가 *분류형* 일 때만 가능 , load_boston 데이터는 회귀형 데이터라 안됨.
    
)



#2. 모델구성 # 분류형 모델
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(4,)))
#모델을 늘리는 것도 성능에 큰 차이를 줌
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(3, activation='softmax')) # 이진 모델과 같이 'softmax' 고정

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=1)

                                                              
#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)


# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)


from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1) # y_predict 가장 큰 값의 자릿수 뽑음 : 예측한 값

print( 'y_predict(예측값)' , y_predict)

y_test = np.argmax(y_test, axis=1) 

# y_test : y_test 값, (원 핫 인코딩을 진행했기 때문에, 다시 원복) 

print( 'y_test(원래값)' , y_test)

acc = accuracy_score(y_test, y_predict) 

print(acc)

