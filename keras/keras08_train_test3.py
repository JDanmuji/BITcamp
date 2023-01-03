import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


# #1. 데이터
x = np.array([1, 2, 3, 4 ,5 ,6, 7, 8, 9, 10]) # (10, )
y = np.array(range(10))                       # (10, )

# [검색] train과 test를 섞어서 7:3으로 맹그러!!
# 힌트 : 사이킷런

#random_state=0 난수 seed 값 사용 시 난수표에 있는 난수 형태로 나옴
#test_size=0.3 사용 가능하지만 #test_size=0.3,train_size=0.7 이면 error
#suffle default : true
x_train ,x_test , y_train , y_test = train_test_split(x, y, #test_size=3, 
                                                      train_size=7)

print('x_train : ' , x_train)
print('x_test : ' , x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)

'''

[결과]

[10  2  7  8  4  1  6]
[3 9 5]
[9 1 6 7 3 0 5]
[2 8 4]

'''