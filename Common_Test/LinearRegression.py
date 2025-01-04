import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # 设置后端为 TkAgg
import matplotlib.pyplot as plt

# 示例数据
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# 添加偏置项
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加 x0 = 1
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y  # 正规方程解

# 预测
X_new = np.array([[0], [6]])
X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
y_pred = X_new_b @ theta_best

# 绘图
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X_new, y_pred, color="red", label="Prediction")
plt.legend()
plt.show()
