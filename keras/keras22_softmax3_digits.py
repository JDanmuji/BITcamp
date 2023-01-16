import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape)
print(np.unique(y, return_counts=True))

print(y)

y = to_categorical(y)
print('===============================================================')

print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    shuffle=True, # False 의 문제점은 데이터 분할 시, 한쪽으로 쏠림 현상 발생으로 데이터의 훈련도의 오차가 심해진다. 
    random_state=333, # random_state 를 사용 시, 분리된 데이터가 비율이 안맞는 현상 발생
    test_size=0.3,
    stratify=y # 분리된 데이터가 비율이 일정하게 됨, 데이터 자체(y)가 분류형 일 때만 가능 , load_boston 데이터는 회귀형 데이터라 안됨.
    
)


#2. 모델구성 # 분류형 모델
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(64,)))
#모델을 늘리는 것도 성능에 큰 차이를 줌
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='linear'))
model.add(Dense(10, activation='softmax')) # 이진 모델과 같이 'softmax' 고정


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

                                                              
#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)




from sklearn.metrics import accuracy_score
import numpy as np


print (x_test)
print (x_train)
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1) # y_predict 가장 큰 값의 자릿수 뽑음 : 예측한 값

print( 'y_predict(예측값)' , y_predict)

y_test = np.argmax(y_test, axis=1) # y_test : y_test 값, (원 핫 인코딩을 진행했기 때문에, 다시 원복) 

print( 'y_test(원래값)' , y_test)

acc = accuracy_score(y_test, y_predict) 

print(acc)

# # 이미지
# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.image[5])
# plt.show()

