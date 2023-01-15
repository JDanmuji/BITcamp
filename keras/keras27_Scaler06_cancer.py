import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 텐서플로 2.8부터는 다시 keras 분리, 경고줄 떠도 실행가능
from tensorflow.keras.callbacks import EarlyStopping #파이썬 클래스 대문자로 시작   
from sklearn.metrics import r2_score,accuracy_score
import numpy as np 

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']


x_train, x_test, y_train, y_test = train_test_split (
    x, y, shuffle=True, random_state=333, test_size=0.2
)


# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델 구성 # 이진 모델
model =  Sequential()
model.add(Dense(50, activation='linear', input_dim=(30)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
                #이진 데이터 사용할 때 0과 1의 사이 한정하는 sigmoid 이용
model.add(Dense(1, activation='sigmoid'))


import time
              
model.compile(loss='binary_crossentropy', #이진데이터 이용 시 loss 함수는 binary_crossentropy 사용
              optimizer='adam',
              metrics=['accuracy'] # 이진 데이터 사용 시 metrics 함수의 accuracy 사용, metrics를 사용하면 히스토리에 나옴.
              )                                 




#earlyStopping 약점 : 5번을 참고 끊으면 그 순간에 weight가 저장 (끊는 순간)
                                                    
                                                                                               
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=20, #참을성     
                              restore_best_weights=True, #최소값에 했던 지점에서 멈춤
                              verbose=1
                              )

hist = model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[earlyStopping], validation_split=0.2, verbose=1) #fit 이 return 한다.


#3. 평가, 예측
#loss = model.evaluate(x_test, y_test)
loss, accuracy = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
y = y_predict[:10] # 정수형으로 바꿔줘야겠죵?
print(y_test[:10])
#accuracy_score 보면 y_predict(실수), y_test(이진코드) 자료형이 안맞음

y = list(map(int, y))
print(y)

# 자료형 변환
y_predict = list(map(int, y_predict))
y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)

print(y_predict)
print(list(map(int, y_predict[:10])))


print('============================================')
print(hist) # <keras.callbacks.History object at 0x00000258175F20A0>
print('============================================')
print(hist.history) # loss, vel-loss 의 변화 형태(딕셔너리 형태|key-value) , value의 형태가 list
print('============================================')
print(hist.history['loss'])
print('============================================')
print(hist.history['val_loss'])
print('============================================')
print('loss : ', loss, ' accuracy : ', accuracy )
print('============================================')
print(' accuracy_score : ', acc )
print('============================================')


plt.figure(figsize=(9,6))
# x 명시 안해도 됨
# hist loss 사용, 색은 red, 선모양은 ., y 선의 이름은 loss
#학습 이력(History) 정보를 리턴
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# hist val_loss 사용, 색은 blue, 선모양은 ., x 선의 이름은 val_loss
plt.plot(hist.history['val_loss'], c='blue',  marker='.' , label='val_loss' )
# 차트 gird 생성
plt.grid() 
# x 축 이름 
plt.xlabel('epochs')
# y 축 이름 
plt.ylabel('loss')
# 차트 제목
plt.title('boston loss')
# 그래프 선 이름 표
plt.legend()
#plt.legend(loc='upper right')  그래프 선 이름 표, 위치
# 차트 창 띄우기
plt.show()




'''

[Scaler 변경 전]

============================================
loss :  0.15744878351688385  accuracy :  0.9385964870452881
============================================
 accuracy_score :  0.38596491228070173
============================================


[Scaler 변경 후]

============================================
loss :  0.22555291652679443  accuracy :  0.9561403393745422
============================================
 accuracy_score :  0.6403508771929824
============================================

'''