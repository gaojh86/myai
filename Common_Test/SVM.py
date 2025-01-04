import numpy as np
import matplotlib.pyplot as plt

# SVM 梯度下降
def svm_gradient_descent(X, y, lr=0.01, epochs=1000, lambda_param=0.01):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    for _ in range(epochs):
        for i in range(m):
            if y[i] * (X[i] @ w + b) < 1:
                w -= lr * (lambda_param * w - y[i] * X[i])
                b -= lr * -y[i]
            else:
                w -= lr * lambda_param * w
    return w, b

# 示例数据
X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
y = np.array([-1, -1, 1, 1])

# 训练 SVM
w, b = svm_gradient_descent(X, y)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', label="Training data")
x_boundary = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
y_boundary = -(w[0] * x_boundary + b) / w[1]
plt.plot(x_boundary, y_boundary, 'g-', label="Decision boundary")

# 预测点
X_test = np.array([[1, 3], [4, 5]])
predictions = np.sign(X_test @ w + b)
plt.scatter(X_test[:, 0], X_test[:, 1], c='yellow', edgecolor='black', s=100, label="Test points")
plt.legend()
plt.show()

print("Predictions:", predictions)
