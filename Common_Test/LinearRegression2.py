import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 设置中文字体（以SimHei为例，需要确保该字体已安装）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 处理负号的显示问题

# 创建示例数据集
data = {
    '面积': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    '价格': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}
df = pd.DataFrame(data)

# 分割特征和目标
X = df[['面积']]
y = df['价格']

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型并训练
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算误差
mse = mean_squared_error(y_test, y_pred)
print(f'均方误差: {mse:.2f}')

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='真实值', s=100)  # 绘制真实值
plt.plot(X_test, y_pred, color='red', linewidth=2, label='预测值')  # 绘制预测值
plt.title('线性回归结果')
plt.xlabel('面积 (平方英尺)')
plt.ylabel('价格 (美元)')
plt.legend()
plt.grid()
plt.show()