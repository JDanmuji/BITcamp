import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. 데이터
x1 = np.array([[1, 2, 3]])
x2 = np.array([[1, 2], [3, 4 ], [5, 6]])
x3 = np.array([[[1, 2]]])
x4 = np.array([[1], [2],[3]])
# 5번 틀림
x5 = np.array([[[1], [2]]])
# x6 = np.array([[2, 3, 4], [5, 6, 7], [8, 9]])
x7 = np.array([[1, 2]])
# 8번 틀림
x8 = np.array([[[1, 2], [3, 4]]])
# x9 = np.array([[[1, 2], [3], [4, 5]]])
x10 = np.array([[1], [2], [3], [4], [5]])


print(x1.shape)
print(x2.shape)
print(x3.shape)
print(x4.shape)
print(x5.shape)
print('    ')
print(x7.shape)
print(x8.shape)
print('    ')
print(x10.shape)
