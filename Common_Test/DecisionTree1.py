import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier, plot_tree  
from sklearn.metrics import accuracy_score

# 设置中文字体（以SimHei为例，需要确保该字体已安装）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 处理负号的显示问题

# 加载数据集
iris = load_iris()  
X = iris.data  
y = iris.target  

# 拆分数据集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  

# 创建决策树模型并训练  
model = DecisionTreeClassifier()  
model.fit(X_train, y_train)

# 预测  
y_pred = model.predict(X_test)  

# 计算准确率  
accuracy = accuracy_score(y_test, y_pred)  
print(f'准确率: {accuracy:.2f}')  

# 绘制决策树  
plt.figure(figsize=(12, 8))  
plot_tree(model, filled=True)  
plt.title('决策树模型')  
plt.show()