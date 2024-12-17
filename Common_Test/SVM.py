import numpy as np

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
X = np.array([[1, 2], [2, 3], [3, 3], [5, 6]])
y = np.array([-1, -1, 1, 1])  # 标签为 -1 或 1

# 训练 SVM
w, b = svm_gradient_descent(X, y)

# 预测
X_test = np.array([[1, 3], [4, 5]])
predictions = np.sign(X_test @ w + b)
print("Predictions:", predictions)
