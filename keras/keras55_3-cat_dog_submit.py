# 개 사진 고양이 사진 한 개를 인터넷에서 잘라내서 뭔지 맞추기
from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np
import cv2 
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout
from keras.models import load_model
from skimage.transform import resize



imgFile = 'C:/Users/bitcamp/Desktop/dog/cat.jpg'

im = cv2.imread(imgFile) #사진 읽어들이기
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) #색공간 변환
im = resize(im, (128,128,3)) #사이즈 조정

#이미지 출력

import matplotlib.pyplot as plt
plt.imshow(im)
plt.show()

print(im)

from keras.models import Sequential
from keras.layers import Dense

#입력 데이터 크기 : 32*32 픽셀, RGB형 이미지 데이터 (데이터 전처리 포스팅 참고)
in_size = 100*100*3 

#출력 데이터 크기 : 10개의 카테고리
num_classes=2

print(im.shape)
model = Sequential()
#입력층 생성
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
#출력층 생성
model.add(Dense(num_classes,activation='softmax'))

path = './_save/'
model = load_model(path + 'cat_dog_model1.h5')

labels = ["cat", "dog"]


r = model.predict(im.reshape(1, 128, 128, 3))
res = r[0]

for i, acc in enumerate(res) :
    print(labels[i], "=", int(acc*100))
print("---")
print("예측한 결과 = " , labels[res.argmax()])