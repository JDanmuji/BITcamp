import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping #파이썬 클래스 대문자로 시작   
from sklearn.metrics import r2_score,accuracy_score
import numpy as np 

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

# print(datasets)
# print(datasets.DESCR) # 컬럼 정보
# print(datasets.feature_names) 
#print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split (
    x, y, shuffle=True, random_state=333, test_size=0.2
)

#2. 모델 구성
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
                              restore_best_weights=True, 
                              verbose=1
                              )

hist = model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[
                 earlyStopping], validation_split=0.2, verbose=1)  # fit 이 return 한다.


#3. 평가, 예측
#loss = model.evaluate(x_test, y_test)
#loss, accuracy = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
#accuracy_score 보면 y_predict(실수), y_test(이진코드) 자료형이 안맞음

# 자료형 변환
y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)

print(y_predict)


