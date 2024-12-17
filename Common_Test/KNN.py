import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# KNN 实现
def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for x in X_test:
        distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
        k_indices = distances.argsort()[:k]
        k_labels = y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)

        # 图形化展示
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', label="Training data")
        plt.scatter(x[0], x[1], c='yellow', edgecolor='black', s=100, label="Test point")
        for i in k_indices:
            plt.plot([x[0], X_train[i, 0]], [x[1], X_train[i, 1]], 'k--', alpha=0.6)
        plt.legend()
        plt.show()

    return np.array(predictions)


# 示例数据
X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[2, 3]])

# 预测
predictions = knn_predict(X_train, y_train, X_test, k=3)
print("Predictions:", predictions)
