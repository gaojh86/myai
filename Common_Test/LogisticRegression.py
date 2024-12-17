import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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
y = np.array([0, 0, 1, 1])  # 二分类

# 添加偏置项
X_b = np.c_[np.ones((X.shape[0], 1)), X]
theta = logistic_regression(X_b, y)

# 预测
X_test = np.array([[1, 2], [4, 5]])
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
predictions = sigmoid(X_test_b @ theta) >= 0.5
print("Predictions:", predictions)
