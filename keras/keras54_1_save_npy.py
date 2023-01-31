import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#keras53 : 이미지를 학습시는 방법 들


# 이미지를 변환하고 증폭시키는 역할
xy_train = ImageDataGenerator(
    rescale=1./255, # 원본을 수치화 한 것만 가지고 있음
    # horizontal_flip=True, # 이미지 수평으로 
    # vertical_flip=True,
    # width_shift_range=0.1, #이동 
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest' # 가까이 있는 것으로 채움
)

xy_test= ImageDataGenerator(
      rescale=1./255
)
                        #파일
xy_train = xy_train.flow_from_directory( # 안에 있는 ad, normal 은 0, 1로 인식
    './_data/brain/train/', 
    target_size=(200,200), #크기에 상관없이 200, 200 을 압축
    batch_size=100000, # 모든 데이터를 가지고 오기
    class_mode='categorical', #수치
    color_mode='grayscale',
    shuffle=True
    # Found 160 images belonging to 2 classes.
) 

                     #파일
xy_test = xy_test.flow_from_directory( # 안에 있는 ad, normal 은 0, 1로 인식
    './_data/brain/test/', 
    target_size=(200,200), #크기에 상관없이 200, 200 을 압축
    batch_size=10000,  #batch_size를 최대한 늘려서 데이터를 한번에 뽑아낼 수 있음
    class_mode='categorical', #수치
    color_mode='grayscale',
    shuffle=True
    # Found 120 images belonging to 2 classes.
) 

print(xy_train)
# <keras.preprocessing.image.ImageDataGenerator object at 0x0000012E32072FA0>

# from sklearn.datasets import load_iris
# datasets = load_iris()

print(xy_train[0][1])
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])
#np.save('./_data/brain/brain_xy_train.npy', arr=xy_train[0])

np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])


