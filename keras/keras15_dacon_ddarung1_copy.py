import numpy as np 
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# [실습]
# dropna 설명, RMSE 튜닝
# dacon
#주어진 데이터를 바탕으로 따릉이 대여량을 예측 해보세요!

# 1. 데이터
path = './_data/ddarung/'                    
                                            #index 컬럼은 0번째
train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # [715 rows x 9 columns]
test_csv = pd.read_csv(path + 'test.csv', index_col=0)     #[1459 rows x 10 columns]
submission = pd.read_csv(path + 'submission.csv', index_col=0)  #[715 rows x 1 columns], 2개중 count 컬럼을 제외한 나머지 1개


# 결측치 처리 
# 1. 결측치 제거 - 데이터 10%를 지웠기 때문에 좋은 방법은 아님
#test_csv = test_csv.dropna()
# 2. 선형 방법을 이용하여 결측치
#train_csv = train_csv.interpolate(method='linear', limit_direction='forward')
# 3. 0으로 대체
train_csv = train_csv.fillna(0)

x = train_csv.drop(['count'], axis=1) # 10개 중 count 컬럼을 제외한 나머지 9개만 inputing
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                                                                       
)

# print(x_train.shape, x_test.shape)  #(929, 9) (399, 9)
# print(y_train.shape)  #  (929, ) 
# print(submission.shape)  # (715, 1)



#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=9, activation='relu'))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(24))
model.add(Dense(15))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일,
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train , epochs=200, batch_size=32)


#4. 예측, 평가
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

#제출할 놈
# to.csv() 를 사용해서 
# submission_0105.csv를 완성하시오.
y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + 'submission_0105.csv')



print("===================================")
print(y_test)
print(y_predict)
print("submit : ", y_submit) 
print("R2 : " , r2)
print("RMSE : ", RMSE(y_test, y_predict)) #54.9284510177609
print("===================================")

