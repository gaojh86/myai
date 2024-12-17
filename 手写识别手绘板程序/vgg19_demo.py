import h5py
import numpy as np
import time
import VGG19
import function
import matplotlib.pyplot as plt
from PIL import Image
# 制作数据集合
# train_image_path = './MNIST/train-images-idx3-ubyte/train-images.idx3-ubyte'
# train_lable_path = './MNIST/train-labels-idx1-ubyte/train-labels.idx1-ubyte'
# teat_image_path = './MNIST/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
# teat_lable_path = './MNIST/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'
#
# train_image = function.jiexi_image(train_image_path)
# train_lable = function.jiexi_label(train_lable_path)
# teat_image = function.jiexi_image(teat_image_path)
# test_lable = function.jiexi_label(teat_lable_path)


# train_image_rgb ,train_lable_rgb,teat_image_rgb,test_lable_rgb = VGG19.HDToRGB(train_image,train_lable,teat_image,test_lable)
#
# VGG19.mnistToH5(train_image_rgb,train_lable_rgb,teat_image_rgb,test_lable_rgb)

def load_data(path):
    f = h5py.File(path,"r")
    x0 = f["train_image"][:]
    y0 = f["train_label"][:]
    x1 = f["test_image"][:]
    y1 = f["test_label"][:]
    f.close()
    return x0,y0,x1,y1

train_image,train_label,test_image,test_label = load_data("./MNISTH5/MNIST.h5")
# print(train_image.shape,"\n",train_label.shape,"\n",test_image.shape,"\n",test_label.shape)


# VGG19的说明
# 19 的由来：
#     一共有19层，包括16层卷积层，最后的3层全连接岑层
#  池化层采用maxpool最大池化层 p =0，s = 2，2*2
#  最终经过softmax
#  3*3 的卷积核 通道数待查

times = 1000
# for time in range(times):
#     #层输入x
#     x = 0
f_h1 = {"pad":1,"stride":1}
f_w1 = np.random.randn(3,3,3,64)
f_b1 = np.random.randn(1,1,1,64)

f_h2 = {"pad":1,"stride":1}
f_w2 = np.random.randn(3,3,3,64)
f_b2 = np.random.randn(1,1,1,64)

p_h1 = {"f":2,"stride":2}



X0 = train_image[0:1000]
a = time.time()
X1,chche_conv1 = VGG19.conv_forward(X0,f_w1,f_b1,f_h1)
# X2,chche_conv2 = VGG19.conv_forward(X1,f_w2,f_b2,f_h2)
# X_pooled1 = VGG19.pool_forword(X2,p_h1)
b = time.time()
print("1cen :",b-a)





















print("down")



