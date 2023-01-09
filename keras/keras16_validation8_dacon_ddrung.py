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


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train , epochs=200, batch_size=32 , validation_data=(x_validation, y_validation)
          )


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
print("===================================")

