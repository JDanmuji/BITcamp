import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)]) # 0 ~ (x-1), 0~9
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]])


#[실습] train_test_split를 이용하여 7:3으로 잘라서 모델 구현/소스 완성

x_train ,x_test = train_test_split(x, test_size=0.3)
y_train , y_test = train_test_split(x, test_size=0.3)

print('x_train : ' , x_train)
print('x_test : ' , x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)







