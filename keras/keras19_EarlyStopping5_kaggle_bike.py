import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping #파이썬 클래스 대문자로 시작   
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



#1. 데이터
path = './_data/bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

#train_csv = train_csv.interpolate(method='linear', limit_direction='forward')
x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']

x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    train_size=0.85, test_size=0.15, shuffle=True
)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
   test_size=0.2, shuffle=True
)


#2. 모델 구성
inputs = Input(shape=(8,))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128, activation='relu') (hidden1)
hidden3 = Dense(64, activation='relu') (hidden2)
hidden4 = Dense(32, activation='relu') (hidden3)
hidden5 = Dense(16, activation='relu') (hidden4)
hidden6 = Dense(8) (hidden5)
output = Dense(1) (hidden6)

model = Model(inputs=inputs, outputs=output)




#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')                                 
start = time.time()                                                                                                                     

earlyStopping = EarlyStopping(monitor='val_loss',  #학습 조기종료를 위해 관찰하는 항목(Default : val_loss)    
                              mode='auto',  #관찰항목에 대해 개선이 없다고 판단하기 위한 기준
                              patience=1000, #참을성, 파라미터를 사용하여 val_loss가 1번 증가할 때 학습을 멈추지 않고 5번의 에포크를 기다리고 종료하는 모델    
                              restore_best_weights=True, # True라면 training이 끝난 후, model의 weight를 monitor하고 있던 값이 가장 좋았을 때의 weight로 복원
                                                         #  False라면, 마지막 training이 끝난 후의 weight로 놔둡니다.
                              verbose=1    #1일 경우, EarlyStopping이 적용될 때, 화면에 적용되었다고 나타냅니다.
                                           #0일 경우, 화면에 나타냄 없이 종료합니다.
                              )

hist = model.fit(x_train, y_train, epochs=10000, batch_size=32, 
                 callbacks=[earlyStopping], #위에서 정의한 콜백함수 넣기
                 validation_split=0.2, 
                 verbose=1
                ) 
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

#제출
y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + 'submission_0106.csv')


print('============================================')
print(hist) # <keras.callbacks.History object at 0x00000258175F20A0>
print('============================================')
print(hist.history) # loss, vel-loss 의 변화 형태(딕셔너리 형태|key-value) , value의 형태가 list
print('============================================')
print(hist.history)
print('============================================')
print(hist.history['loss'])
print('============================================')
print(hist.history['val_loss'])
print('============================================')
print('loss : ', loss)
print('============================================')
print('걸린시간 : ', end - start)


plt.figure(figsize=(9,6))


# x 명시 안해도 됨
# hist loss 사용, 색은 red, 선모양은 ., y 선의 이름은 loss
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
[verbose = 0]
loss :  40.98577880859375
걸린시간 :  9.872689485549927


[verbose = 1]
loss :  45.78962326049805
걸린시간 :  11.606598138809204

[verbose = 2] 
loss :  45.07732009887695
걸린시간 :  10.344038963317871


'''