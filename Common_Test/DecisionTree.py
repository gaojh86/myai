import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, depth=0, max_depth=3):
        self.depth = depth
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = None

    def fit(self, X, y):
        if len(set(y)) == 1 or self.depth >= self.max_depth:
            self.label = Counter(y).most_common(1)[0][0]
            return
        m, n = X.shape
        best_gain = -1
        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_indices = X[:, feature] <= t
                right_indices = ~left_indices
                gain = self._information_gain(y, left_indices, right_indices)
                if gain > best_gain:
                    best_gain = gain
                    self.feature_index = feature
                    self.threshold = t
        if best_gain == -1:
            self.label = Counter(y).most_common(1)[0][0]
            return
        left_indices = X[:, self.feature_index] <= self.threshold
        right_indices = ~left_indices
        self.left = DecisionTree(self.depth + 1, self.max_depth)
        self.right = DecisionTree(self.depth + 1, self.max_depth)
        self.left.fit(X[left_indices], y[left_indices])
        self.right.fit(X[right_indices], y[right_indices])

    def predict(self, X):
        if self.label is not None:
            return self.label
        if X[self.feature_index] <= self.threshold:
            return self.left.predict(X)
        else:
            return self.right.predict(X)

    def _information_gain(self, y, left_indices, right_indices):
        def entropy(y):
            counts = np.bincount(y)
            probabilities = counts / len(y)
            return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

        parent_entropy = entropy(y)
        n = len(y)
        left_entropy = entropy(y[left_indices])
        right_entropy = entropy(y[right_indices])
        weighted_entropy = (len(y[left_indices]) / n) * left_entropy + (len(y[right_indices]) / n) * right_entropy
        return parent_entropy - weighted_entropy

# 示例数据
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

# 训练决策树
tree = DecisionTree(max_depth=2)
tree.fit(X, y)

# 预测
X_test = np.array([3, 4])
prediction = tree.predict(X_test)
print("Prediction:", prediction)
