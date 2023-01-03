import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


# #1. 데이터
x = np.array([1, 2, 3, 4 ,5 ,6, 7, 8, 9, 10]) # (10, )
y = np.array(range(10))                       # (10, )

# [검색] train과 test를 섞어서 7:3으로 맹그러!!
# 힌트 : 사이킷런


x_train ,x_test , y_train , y_test = train_test_split(x,y, test_size=3, train_size=7, random_state=0)

print(x_train)
print(x_test)
print(y_train)
print(y_test)


# [결과]

# [10  2  7  8  4  1  6]
# [3 9 5]
# [9 1 6 7 3 0 5]
# [2 8 4]
