import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist

#

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() #교육용 자료, 이미 train/test 분류

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) reshape (훈련)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,) (테스트)

print(x_train[1000])
print(y_train[1000])

import matplotlib.pyplot as plt
plt.imshow(x_train[10], 'gray')
plt.show()

