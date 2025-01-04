import numpy as np
import matplotlib.pyplot as plt

# Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 逻辑回归
def logistic_regression(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(epochs):
        z = X @ theta
        h = sigmoid(z)
        gradient = X.T @ (h - y) / m
        theta -= lr * gradient
    return theta

# 示例数据
X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
y = np.array([0, 0, 1, 1])

# 添加偏置项
X_b = np.c_[np.ones((X.shape[0], 1)), X]
theta = logistic_regression(X_b, y)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', label="Training data")
x_boundary = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
y_boundary = -(theta[0] + theta[1] * x_boundary) / theta[2]
plt.plot(x_boundary, y_boundary, 'g-', label="Decision boundary")

# 预测点
X_test = np.array([[2, 3], [4, 5]])
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
predictions = sigmoid(X_test_b @ theta) >= 0.5
plt.scatter(X_test[:, 0], X_test[:, 1], c='yellow', edgecolor='black', s=100, label="Test points")
plt.legend()
plt.show()

print("Predictions:", predictions)
