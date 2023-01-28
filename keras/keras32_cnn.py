from tensorflow.keras.models import Sequential
#이미지는 Conv2D 사용, Flatten 는 tensor 모양의 데이터를 쫙 펴줌
from tensorflow.keras.layers import Dense, Conv2D, Flatten


model = Sequential()

                            # input(60000, 5, 5, 1) => 60000은 손글씨예제
model.add(Conv2D(filters=10, kernel_size=(2, 2), 
                input_shape=(5, 5, 1))) # 행무시 열우선  # (Non(데이터의 개수, 몇 개를 넣어도 상관없기에 n(non) 입력), 4, 4, 10) 
                                                         # (batch_size=훈련의 개수, rows=데이터의 개수, 통배치(전체), columns, channels, )




model.add(Conv2D(5, (2, 2)))
#model.add(Conv2D(filters=5, kernel_size=(2, 2)))         # (N, 3, 3, 5)
#filters 를 조절하여 성능을 높인다.

model.add(Flatten())                                     # (N, 45) , 연산 없음
model.add(Dense(10))                                    # (N, 10)
# 인풋은 (batch_size, input_dim )
model.add(Dense(4, activation='relu'))                   # 지현, 성환, 건률, 렐루 (N, 1)
model.add(Dense(1))                                        # (N, 1)

model.summary()


