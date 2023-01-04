import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# #1. 데이터
x = np.array([1, 2, 3, 4 ,5 ,6, 7, 8, 9, 10]) # (10, )
y = np.array(range(10))                       # (10, )

# 실습 : 넘파이 리스트 슬라이싱!! 7:3으로 잘라라!!!


# 단순히 7:3으로만 나누는 것 데이터가 뭉텅이로 나가기 때문에 좋지 않다.
# 전체 데이터 셋에서 간격에 맞춰서 빼준다. 데이터 영향 안가게

x_train = x[:7]
x_test = x[7:]
y_train = y[:7]
y_test = y[7:]
# x_train = x[:-3] 
# x_test = x[-3:]
# y_train = y[:-3]
# y_test = y[-3:]

print(x_train)
print(x_test)
print(y_train)
print(y_test)

