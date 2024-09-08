import numpy as np
from sklearn import datasets
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score


# 加载鸢尾花数据集
iris = datasets.load_iris()

data = iris.data
scaler = StandardScaler()
print(data[0:2])
data = scaler.fit_transform(data)
num_clusters = 3  # 目标聚类数目
print(data[0:2])

# X = data[:, [2, 3]]
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
 
# # 设置画布尺寸
# plt.figure(figsize=(10, 6))
 
# # s: 正方形 x: x型 o: 圆形
# markers = ('s', 'x', 'o')
# colors = ('red', 'blue', 'lightgreen')
# classes = ('Setosa', 'Versicolour', 'Virginica')
# cmap = ListedColormap(colors[:len(np.unique(y_test))])
# for idx, cl in enumerate(np.unique(y)):
#     plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], color=cmap(idx),
#                 marker=markers[idx], label=classes[cl])
 
# # 创建图例
# plt.legend()
# plt.show()

def calculate_fitness(individual, data):
    # centroids = np.array(individual).reshape(-1, data.shape[1])
    # labels = np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)
    
    # # 计算轮廓系数和Calinski-Harabasz得分
    # silhouette = silhouette_score(data, labels)
    # calinski_harabasz = calinski_harabasz_score(data, labels)
    
    # # 返回负值，因为NSGA-II是最小化问题
    # return -silhouette, -calinski_harabasz
    
    
    centroids = individual.reshape((num_clusters, -1))
    total_intra_cluster_distance = 0
    min_inter_cluster_distance = np.inf
    
    for point in data:
        distances = np.linalg.norm(point - centroids, axis=1)
        total_intra_cluster_distance += np.min(distances)

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < min_inter_cluster_distance:
                min_inter_cluster_distance = dist
                
    return total_intra_cluster_distance, -min_inter_cluster_distance
    
    # centroids = individual.reshape((num_clusters, -1))
    # distances = np.array([np.linalg.norm(X - center, axis=1) for center in centroids])
    # min_distance = np.min(distances, axis=0)
    # xb = np.sum(np.sum(distances ** 2, axis=1)) / (X.shape[0] * np.min(distances ** 2))
    
    # return -xb,  

# 创建适应度类和个体类
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

def initialize_individual(icls):
    return icls(np.random.rand(num_clusters * data.shape[1]))

toolbox = base.Toolbox()
toolbox.register("individual", initialize_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", calculate_fitness, data=data)

def main():
    population = toolbox.population(n=100)
    ngen = 100  # 迭代次数
    cxpb = 0.9  # 交叉概率
    mutpb = 0.1  # 变异概率

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                              stats=stats, verbose=True)
    

    # 使用 NSGA-II 进行排序
    # algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=cxpb, mutpb=mutpb, ngen=ngen, 
    #                           stats=stats, halloffame=None, verbose=True)

    return population

if __name__ == "__main__":
    final_population = main()

    # 输出最终的非支配解
    front = tools.sortNondominated(final_population, len(final_population), first_front_only=True)[0]
    for individual in front:
        print(individual.fitness.values)