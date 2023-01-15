import time
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping #파이썬 클래스 대문자로 시작   

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'                    
                                            #index 컬럼은 0번째
train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # [715 rows x 9 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)     #[1459 rows x 10 columns]
submission = pd.read_csv(path + 'submission.csv', index_col=0)  #[715 rows x 1 columns], 2개중 count 컬럼을 제외한 나머지 1개

# 결측치 처리 
# 1. 선형 방법을 이용하여 결측치
train_csv = train_csv.interpolate(method='linear', limit_direction='forward')

x = train_csv.drop(['count'], axis=1) # 10개 중 count 컬럼을 제외한 나머지 9개만 inputing
y = train_csv['count']


x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    train_size=0.85, test_size=0.15, shuffle=False
)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
    test_size=0.3, shuffle=False
)

#2. 모델 구성
inputs = Input(shape=(9, ))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128, activation='relu') (hidden1)
hidden3 = Dense(64, activation='relu') (hidden2)
hidden4 = Dense(32) (hidden3)
hidden5 = Dense(16) (hidden4)
hidden6 = Dense(8) (hidden5)
output = Dense(1) (hidden6)


model = Model(inputs=inputs, outputs=output)



model.compile(loss='mae', optimizer='adam')                                 
start = time.time()                                                                         

#earlyStopping 약점 : 5번을 참고 끊으면 그 순간에 weight가 저장 (끊는 순간)
                                                    
                                                                                               
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='auto', 
                              patience=1000, #참을성     
                              restore_best_weights=True, 
                              verbose=1
                              )

hist = model.fit(x_train, y_train, epochs=10000, batch_size=32, callbacks=[earlyStopping], validation_split=0.2, verbose=1) #fit 이 return 한다.
end = time.time()

#3. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

#제출
y_submit = model.predict(test_csv)

# to.csv() 를 사용해서 submission_0105.csv를 완성하시오.
submission['count'] = y_submit
submission.to_csv(path + 'submission_0105.csv')


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