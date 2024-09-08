import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from FCM2 import FCM 
from scipy.spatial.distance import cdist


def calculate_Jm2(data, centroids, U, m):
    Um = U ** m
    Jm = np.sum(Um * np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)**2)
    return Jm

def calculate_XB2(data, centroids, U, m):
    n_samples = data.shape[0]
    Um = U ** m
    min_intercluster_dist = np.min([np.linalg.norm(centroids[i] - centroids[j])**2 
                                    for i in range(len(centroids)) for j in range(len(centroids)) if i != j])
    sum_intracluster_dist = np.sum(Um * np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)**2)
    XB = sum_intracluster_dist / (n_samples * min_intercluster_dist)
    return XB

def calculate_Jm(centroids, labels, data, u):
    Jm = 0.0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        print(np.linalg.norm(cluster_points - centroid, axis=1).shape)
        Jm += np.sum(np.linalg.norm(cluster_points - centroid, axis=1)**2)
    return Jm

def calculate_XB(centroids, labels, data, u):
    min_intercluster_dist = np.inf
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < min_intercluster_dist:
                min_intercluster_dist = dist

    sum_intracluster_dist = 0.0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        sum_intracluster_dist += np.sum(np.linalg.norm(cluster_points - centroid, axis=1)**2)

    XB = sum_intracluster_dist / (len(data) * min_intercluster_dist**2)
    return XB

# 定义适应度函数
def evaluate(individual, data):
    # 将个体的基因转换为聚类中心
    centroids = np.array(individual).reshape(-1, data.shape[1])
    
    # 计算所有样本基于当前聚类中心centroids的标签值（类别值0,1,2）
    # labels[i] == 0 -> centroids[0]
    # lables[i] == 1 -> centroids[1]
    # lables[i] == 2 -> centroids[2]
    # len(labels) == 150
    # len(data) == 150
    # labels[i]就是data[i]（第i+1个鸢尾花数据）的类别值
    # labels = np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)
    # print(labels)
    
    # 更新隶属度矩阵
    m = 2
    dist = cdist(data, centroids)
    u = 1 / (dist ** (2 / (m - 1)))
    u = u / u.sum(axis=1, keepdims=True)
    
    # 计算 Jm 和 XB 指标
    # Jm = calculate_Jm(centroids, labels, data, u)
    # XB = calculate_XB(centroids, labels, data, u)
    
    Jm = calculate_Jm2(data, centroids, u, m)
    XB = calculate_XB2(data, centroids, u, m)
    
    return Jm, XB

def demo1():
    iris = load_iris()
    data = iris.data
    print(data[:, None].shape)
    # labels = np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)

def main():
    # 设置遗传算法参数
    POP_SIZE = 100
    NGEN = 200
    CXPB = 0.7      # 交叉概率,即个体进行交叉操作的概率。取值范围为 [0, 1]。
    MUTPB = 0.2     # 变异概率,即个体进行变异操作的概率。取值范围为 [0, 1]。
    N_CLUSTERS = 3
    np.random.seed(42)
    
    # 加载鸢尾花数据集
    iris = load_iris()
    data = iris.data
    # print(f"data shape:{data.shape}")
    
    # 归一化数据
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # print(data[0:5])

    # 创建适应度类 -1.0表示最小化 1.0表示最大化
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0)) # 负值表示最小化
    # 个体是个list 长度为12
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    # 注册相关函数
    toolbox = base.Toolbox()
    
    # 注册个体初始化函数
    toolbox.register("attr_float", np.random.uniform, np.min(data), np.max(data))
    # val = toolbox.attr_float()
    # print(f"val:{val}")
    
    # 注册个体的长度/初始化方法及细则
    # shape->[150, 4] shape[0]->150 shape[1]->4
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, N_CLUSTERS * data.shape[1])
    # ind = toolbox.individual()
    # print(ind)
    
    # 注册种群函数
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # pop = toolbox.population(n=POP_SIZE)
    # print(pop)
    
    
    # 注册交叉函数 混合交叉
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    # 注册突变函数 有界多项式突变
    toolbox.register("mutate", tools.mutPolynomialBounded, low=np.min(data), up=np.max(data), eta=1.0, indpb=0.2)
    # 注册选择函数
    toolbox.register("select", tools.selNSGA2)
    # 注册评估函数
    toolbox.register("evaluate", evaluate, data=data)

    # 初始化种群
    population = toolbox.population(n=POP_SIZE)
    # print(population[0:3])
    
    # 初始化隶属度矩阵
    pop_u = np.random.rand(POP_SIZE, data.shape[0], data.shape[1])
    print(pop_u.shape)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # 运行遗传算法
    result_pop, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, verbose=True, stats=stats)

    Jm_XB = []
    plt.figure(figsize=(8, 6))
    for ind in result_pop:
        Jm, XB = toolbox.evaluate(ind)
        plt.scatter(Jm, XB)
    plt.show()


    # 获取最优个体
    top_individuals = tools.selBest(result_pop, k=2)
    best_individual = top_individuals[0]

    # 提取聚类中心
    best_centroids = np.array(best_individual).reshape(-1, data.shape[1])
    labels = np.argmin(np.linalg.norm(data[:, None] - best_centroids, axis=2), axis=1)

    # 可视化结果聚类结果
    plt.figure(figsize=(8, 6))
    for i in range(N_CLUSTERS):
        cluster_data = data[labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}')
    plt.scatter(best_centroids[:, 0], best_centroids[:, 1], s=300, c='red')
    plt.show()

if __name__ == "__main__":
    main()