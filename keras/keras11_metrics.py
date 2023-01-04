from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split


# error 가 영향을 미친다. 그 다음 가중치를 갱신할 때
# loss, 손실함수는 훈련에 영향을 미친다.

#1. 데이터
x = np.array(range(1,21))
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                                                                       
)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
# metrics=['mae'] 는 훈련에 영향을 끼치지 않는다. 하지만 결과를 같이 참고할 수 있다.
# metrics=['mae', 'mse'] 1/1 [==============================] - 0s 94ms/step - loss: 14.7177 - mae: 2.9771 - mse: 14.7177
# loss :  [14.717720031738281, 2.977104425430298, 14.717720031738281]
# metrics=['mae', 'mse', 'accuracy'] accuracy 를 사용할 수 없는 모델이지만, 참고는 할 수 있다. 'accuracy' = 'acc'
            # 가중치를 갱신할 때 사용하는 것들               
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse', 'accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)



'''

    [mae의 결과]
    loss :  3.146573305130005
    
    [mse의 결과]
    loss :  15.29187297821045
    
'''




