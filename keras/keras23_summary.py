from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np 

# 1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])


# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary() # 모델의 아키텍처 구조


'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 5)                 10( (inputnode * outputnode) + output Node(bias))

 dense_1 (Dense)             (None, 4)                 24

 dense_2 (Dense)             (None, 3)                 15

 dense_3 (Dense)             (None, 2)                 8

 dense_4 (Dense)             (None, 1)                 3

=================================================================

'''