import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지를 변환하고 증폭시키는 역할
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 이미지 수평으로 
    vertical_flip=True,
    width_shift_range=0.1, #이동 
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearset' # 가까이 있는 것으로 채움
)

test_datagen= ImageDataGenerator(
      rescale=1./255
)
