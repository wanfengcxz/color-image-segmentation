import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

# 创建示例数据
X = np.random.rand(100, 2)

# 进行 Mean Shift 聚类
mean_shift = MeanShift()
mean_shift.fit(X)

# 获取聚类标签
labels = mean_shift.labels_
# 获取聚类中心
cluster_centers = mean_shift.cluster_centers_
# 获取聚类数量
n_clusters = mean_shift.cluster_centers_.shape[0]

print(f"聚类数量: {n_clusters}")
print(f"聚类中心:\n{cluster_centers}")

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.scatter(
    cluster_centers[:, 0], cluster_centers[:, 1], s=300, c="red", marker="X"
)  # 聚类中心
plt.title("Mean Shift Clustering")
plt.show()
