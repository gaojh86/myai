from sklearn.datasets import make_blobs  
from sklearn.cluster import KMeans  
import matplotlib.pyplot as plt

# 设置中文字体（以SimHei为例，需要确保该字体已安装）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 处理负号的显示问题

# 生成示例数据  
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)  

# 创建 K均值模型  
kmeans = KMeans(n_clusters=4)  
kmeans.fit(X)  
y_kmeans = kmeans.predict(X)  

# 绘制结果  
plt.figure(figsize=(10, 6))  
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')  
centers = kmeans.cluster_centers_  
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, label='聚类中心')  
plt.title("K均值聚类结果")  
plt.xlabel('特征 1')  
plt.ylabel('特征 2')  
plt.legend()  
plt.grid()  
plt.show()