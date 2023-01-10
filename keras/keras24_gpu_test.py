import tensorflow as tf 


print(tf.__version__) # 2.7.4


gpus = tf.config.experimental.list_physical_devices('GPU')

# 2.7.4
# 인터프린터 GPU로 교환
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


print(gpus)

if(gpus): 
    print("GPU 돈다.")
else :
    print("GPU 안돈다.")
    
