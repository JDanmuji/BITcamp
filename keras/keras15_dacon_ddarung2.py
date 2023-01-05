import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# [실습]
# dropna 설명, RMSE 튜닝
# 결측치 나쁜놈!
# 결측치 때문에 To Be Contiunue!

# 1. 데이터
path = './_data/ddarung/'                    
                                            #index 컬럼은 0번째
train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # [715 rows x 9 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)     #[1459 rows x 10 columns]
submission = pd.read_csv(path + 'submission.csv', index_col=0)  #[715 rows x 1 columns], 2개중 count 컬럼을 제외한 나머지 1개



# 결측치 처리 
# 1. 결측치 제거 - 데이터 10%를 지웠기 때문에 좋은 방법은 아님
print(train_csv.isnull().sum()) # null 값만 보임

train_csv = train_csv.dropna()
#test_csv = test_csv.dropna()
print(train_csv.isnull().sum()) # (1328, 10)


x = train_csv.drop(['count'], axis=1) # 10개 중 count 컬럼을 제외한 나머지 9개만 inputing
y = train_csv['count']

print(x) # [1459 rows x 9 columns]
print(y)



x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                                                                       
)

print(x_train.shape, x_test.shape)  #(929, 9) (399, 9)
print(y_train.shape)  #  (929, ) 
print(submission.shape)  # (715, 1)



#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=9, activation='relu'))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(24))
model.add(Dense(15))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x_train, y_train , epochs=100, batch_size=32)
end = time.time()


#4. 예측, 평가
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



print("===================================")
print(y_test)
print(y_predict)
print("submit : ", y_submit) 
print("R2 : " , r2)
print("RMSE : ", RMSE(y_test, y_predict)) 
print("걸린시간 : ", end - start)
print("===================================")




'''
[cpu]
R2 :  0.5837751037336487
RMSE :  52.50501988058989
걸린시간 :  3.5466063022613525

[gpu]
R2 :  0.5786453798085762
RMSE :  52.8275756449218
걸린시간 :  11.391770601272583
'''
