# VGG19的说明
# 19 的由来：
#     一共有19层，包括16层卷积层，最后的3层全连接岑层
#  池化层采用maxpool最大池化层 p =0，s = 2，2*2
#  最终经过softmax
#  3*3 的卷积核 通道数待查

# 一些包
import numpy as np
import h5py
import function
import matplotlib.pyplot as plt
from PIL import Image
# 0填充
def zero_pad(X, pad):
    X_paded = np.pad(X, (
        (0, 0),  # 样本数，不填充
        (pad, pad),  # 图像高度,你可以视为上面填充x个，下面填充y个(x,y)
        (pad, pad),  # 图像宽度,你可以视为左边填充x个，右边填充y个(x,y)
        (0, 0)),  # 通道数，不填充
                     'constant', constant_values=0)  # 连续一样的值填充

    return X_paded
# padding函数的测试
# np.random.seed(1)
# x = np.random.randn(600,28,28,1)
# x_paded = zero_pad(x,1)
# print(x_paded.shape)

#卷积运算
def conv_single_step(X, W,b):

    s = np.multiply(X, W)

    Z = np.sum(s)

    return Z
# 测试conv_single_step
# np.random.seed(1)
# x = np.random.randn(4,4,3)
# w = np.random.randn(4,4,3)
# b = np.random.randn(1,1,1)
# z = conv_single_step(x,w,b)

# 卷积向前
def conv_forward(X,W,b,hparameters):
    pad = hparameters["pad"]
    stride = hparameters["stride"]
    (m,h,w,c_x) = X.shape
    (f,f,n_c_x,f_c) = W.shape
    n_H = int((h - f + 2 * pad) / stride) + 1
    n_W = int((w - f + 2 * pad) / stride) + 1
    Z =np.zeros((m,n_H,n_W,f_c))
    X_pad = zero_pad(X,pad)
    for i in range(m):
        x_pad = X_pad[i]
        for h1 in range(n_H):
            for w1 in range(n_W):
                for c1 in range(f_c):
                    vert_start = h1
                    vert_end = vert_start + f
                    horiz_start = w1
                    horiz_end = horiz_start + f

                    x_temp = x_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # print(x_temp.shape)
                    Z[i, h1, w1, c1] = np.sum(np.multiply(x_temp, W[:, :, :, c1]) + b[0, 0, 0, c1])
    cache = (X, W, b, hparameters)
    return (Z,cache)

# 检测conv_forward
# A_prev = np.random.randn(10,28,28,3)
# W = np.random.randn(3,3,3,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad" : 1, "stride": 1}
# #
# Z , cache_conv = conv_forward(A_prev,W,b,hparameters)

# 池化层
def pool_forword(Z,hparameters,mode ="max"):
    # 获取输入数据的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = Z.shape
    # 获取超参数的信息
    f = hparameters["f"]
    stride = hparameters["stride"]
    # 计算输出维度
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev
    Z_pooled = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):  # 遍历样本
        for h in range(n_H):  # 在输出的垂直轴上循环
            for w in range(n_W):  # 在输出的水平轴上循环
                for c in range(n_C):  # 循环遍历输出的通道
                    # 定位当前的切片位置
                    vert_start = h * stride  # 竖向，开始的位置
                    vert_end = vert_start + f  # 竖向，结束的位置
                    horiz_start = w * stride  # 横向，开始的位置
                    horiz_end = horiz_start + f  # 横向，结束的位置
                    # 定位完毕，开始切割
                    z_slice_prev = Z[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # 对切片进行池化操作
                    if mode == "max":
                        Z_pooled[i, h, w, c] = np.max(z_slice_prev)
                    elif mode == "average":
                        Z_pooled[i, h, w, c] = np.mean(z_slice_prev)
    # 校验完毕，开始存储用于反向传播的值
    cache = (Z, hparameters)
    return Z_pooled,cache

# 检测池化层
# Z = np.random.randn(10,28,28,8)
# hparameters={"f":2,"stride":2}
# A,B = pool_forword(Z,hparameters)

# 卷积反向传播
def conv_backward(dZ, cache):
    """
    实现卷积层的反向传播

    参数：
        dZ - 卷积层的输出Z的 梯度，维度为(m, n_H, n_W, n_C)
        cache - 反向传播所需要的参数，conv_forward()的输出之一

    返回：
        dA_prev - 卷积层的输入（A_prev）的梯度值，维度为(m, n_H_prev, n_W_prev, n_C_prev)
        dW - 卷积层的权值的梯度，维度为(f,f,n_C_prev,n_C)
        db - 卷积层的偏置的梯度，维度为（1,1,1,n_C）

    """
    # 获取cache的值
    (A_prev, W, b, hparameters) = cache

    # 获取A_prev的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 获取dZ的基本信息
    (m, n_H, n_W, n_C) = dZ.shape

    # 获取权值的基本信息
    (f, f, n_C_prev, n_C) = W.shape

    # 获取hparaeters的值
    pad = hparameters["pad"]
    stride = hparameters["stride"]

    # 初始化各个梯度的结构
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # 前向传播中我们使用了pad，反向传播也需要使用，这是为了保证数据结构一致
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    # 现在处理数据
    for i in range(m):
        # 选择第i个扩充了的数据的样本,降了一维。
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # 定位完毕，开始切片
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # 切片完毕，使用上面的公式计算梯度
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        # 设置第i个样本最终的dA_prev,即把非填充的数据取出来。
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    # 数据处理完毕，验证数据格式是否正确
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return (dA_prev, dW, db)

#反向传播测试
# A_prev = np.random.randn(10,28,28,3)
# W = np.random.randn(3,3,3,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad" : 1, "stride": 1}
# Z , cache_conv = conv_forward(A_prev,W,b,hparameters)
# dA , dW , db = conv_backward(Z,cache_conv)
# print("dA_mean =", np.mean(dA))
# print("dW_mean =", np.mean(dW))
# print("db_mean =", np.mean(db))

def create_mask_from_window(x):
    """
    从输入矩阵中创建掩码，以保存最大值的矩阵的位置。

    参数：
        x - 一个维度为(f,f)的矩阵

    返回：
        mask - 包含x的最大值的位置的矩阵
    """
    mask = x == np.max(x)

    return mask


def distribute_value(dz, shape):
    """
    给定一个值，为按矩阵大小平均分配到每一个矩阵位置中。

    参数：
        dz - 输入的实数
        shape - 元组，两个值，分别为n_H , n_W

    返回：
        a - 已经分配好了值的矩阵，里面的值全部一样。

    """
    # 获取矩阵的大小
    (n_H, n_W) = shape

    # 计算平均值
    average = dz / (n_H * n_W)

    # 填充入矩阵
    a = np.ones(shape) * average

    return a


def pool_backward(dA, cache, mode="max"):
    # 获取cache中的值
    (A_prev, hparaeters) = cache

    # 获取hparaeters的值
    f = hparaeters["f"]
    stride = hparaeters["stride"]

    # 获取A_prev和dA的基本信息
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    # 初始化输出的结构
    dA_prev = np.zeros_like(A_prev)

    # 开始处理数据
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    # 选择反向传播的计算方式
                    if mode == "max":
                        # 开始切片
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # 创建掩码
                        mask = create_mask_from_window(a_prev_slice)
                        # 计算dA_prev
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == "average":
                        # 获取dA的值
                        da = dA[i, h, w, c]
                        # 定义过滤器大小
                        shape = (f, f)
                        # 平均分配
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
    # 数据处理完毕，开始验证格式
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev

#检测pool反向
# A_prev = np.random.randn(10,28,28,3)
# W = np.random.randn(3,3,3,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad" : 1, "stride": 1}
# Z , cache_conv = conv_forward(A_prev,W,b,hparameters)
# hparameters={"f":2,"stride":2}
# Z, cache_pool = pool_forword(Z,hparameters)
# DA = pool_backward(Z,cache_pool)
# print(DA.shape)





def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    cache = Z
    return A, cache

# 全连接层
def fc(X,W,b):
    Z = np.dot(W,X)+b
    cache = (X,W,b)
    return Z,cache


# 反向softmax
def softmax_back(Y,Y_out):
    dZ = Y_out-Y
    return dZ

# 全连接向后
def fc_back(dZ,cache):
    Z_prev, W, b = cache
    m = Z_prev.shape[1]
    dW = np.dot(dZ, Z_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dZ_prev = np.dot(W.T, dZ)
    return dZ_prev, dW, db



# 灰度转RGB
def HDToRGB(train_X,train_y,test_X,test_y):
    l0 = train_X.shape[0]
    l1 = test_X.shape[0]
    x0 = []
    x1 = []
    for i in range(l0):
        img = Image.fromarray(train_X[i])
        img = img.convert("RGB")
        x0.append(np.array(img))
    x0 = np.array(x0)
    for i in range(l1):
        img = Image.fromarray(test_X[i])
        img = img.convert("RGB")
        x1.append(np.array(img))
    x1 = np.array(x1)
    y0 = function.label_init(train_y)
    y1 = function.label_init(test_y)
    return x0,y0,x1,y1





def mnistToH5(train_x,train_y,test_x,test_y):
    f= h5py.File("./MNISTH5/MNIST.h5","w")
    f.create_dataset("train_image",data = train_x)
    f.create_dataset("train_label",data = train_y)
    f.create_dataset("test_image",data = test_x)
    f.create_dataset("test_label",data = test_y)
    f.close()

















