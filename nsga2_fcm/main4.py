import numpy as np
import pandas as pd
from sklearn import datasets
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler


# def calculate_fitness(individual, data, num_clusters, num_features):
#     centroids = individual.reshape((num_clusters, num_features))
#     labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
    
#     intra_cluster_distances = np.sum([np.linalg.norm(data[labels == i] - centroids[i], axis=1).sum() for i in range(num_clusters)])
#     inter_cluster_distances = np.sum([np.linalg.norm(centroids[i] - centroids[j]) for i in range(num_clusters) for j in range(i + 1, num_clusters)])
    
#     return intra_cluster_distances, -inter_cluster_distances  # 最小化簇内距离，最大化簇间距离


def calculateXBIndex(membership_mat, centers, m):
    # N, C = membership_mat.shape   # (N,c)
    # print(N, C)
    num_samples = df.shape[0]
    min_distance = float('inf')
    for i in range(len(centers)):
        for j in range(len(centers)):
            if i != j:
                dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                if dist < min_distance:
                    min_distance = dist

    xb_index = 0.0
    # print(membership_mat)
    # print(dataset.shape)
    for i in range(num_samples):
        for j in range(len(centers)):
            xb_index += (membership_mat[i][j] ** m) * (np.linalg.norm(np.array(dataset.iloc[i]) - np.array(centers[j])) ** 2)

    xb_index = xb_index / (num_samples * (min_distance ** 2))
    return xb_index

def J_m(membership_mat, cluster_centers, m):
    # N, C = membership_mat.shape   # (N,c)
    # print(N, C)
    num_samples = df.shape[0]
    min_distance = float('inf')

    Jm_index = 0.0
    # print(membership_mat)
    # print(dataset.shape)
    for i in range(num_samples):
        for j in range(len(cluster_centers)):
            Jm_index += (membership_mat[i][j] ** m) * (np.linalg.norm(np.array(dataset.iloc[i]) - np.array(cluster_centers[j])) ** 2)

    return Jm_index

def calculate_fitness(individual, data, num_clusters, num_features):
    # centroids = np.array(individual).reshape(-1, data.shape[1])
    # labels = np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)
    
    # # 计算轮廓系数和Calinski-Harabasz得分
    # silhouette = silhouette_score(data, labels)
    # calinski_harabasz = calinski_harabasz_score(data, labels)
    
    # # 返回负值，因为NSGA-II是最小化问题
    # return -silhouette, -calinski_harabasz
    
    
    centroids = individual.reshape((num_clusters, -1))
    # total_intra_cluster_distance = 0
    # min_inter_cluster_distance = np.inf
    
    # for point in data:
    #     distances = np.linalg.norm(point - centroids, axis=1)
    #     total_intra_cluster_distance += np.min(distances)

    # for i in range(num_clusters):
    #     for j in range(i + 1, num_clusters):
    #         dist = np.linalg.norm(centroids[i] - centroids[j])
    #         if dist < min_inter_cluster_distance:
    #             min_inter_cluster_distance = dist
                
    # return total_intra_cluster_distance, -min_inter_cluster_distance
    
    centroids = individual.reshape((num_clusters, -1))
    distances = np.array([np.linalg.norm(data - center, axis=1) for center in centroids])
    min_distance = np.min(distances, axis=0)
    xb = np.sum(np.sum(distances ** 2, axis=1)) / (data.shape[0] * np.min(distances ** 2))
    jm =  np.sum(np.sum(distances ** 2, axis=1)) / data.shape[0]
    return -xb, jm

def main():
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    data = iris.data
    num_samples, num_features = data.shape
    num_clusters = 3  # 目标聚类数目

    scaler = StandardScaler()
    print(data[0:2])
    data = scaler.fit_transform(data)
    print(data[0:2])
    # print(data.shape)
    
    ngen = 100  # 迭代次数
    cxpb = 0.9  # 交叉概率
    mutpb = 0.1 # 变异概率


    # 设置多目标优化问题
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    # 创建Individual类，继承np.ndarray
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

    # 生成个体
    toolbox = base.Toolbox()
    toolbox.register("Attr_float", random.uniform, np.min(data), np.max(data))
    toolbox.register("Individual", tools.initRepeat, creator.Individual, toolbox.Attr_float, num_clusters * num_features)
    # ind = toolbox.Individual()
    # print(ind)

    # 生成初始族群
    toolbox.register("Population", tools.initRepeat, list, toolbox.Individual)
    population = toolbox.Population(n=ngen)
    print(population[0:2])

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=0.5, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", calculate_fitness, data=data, num_clusters=num_clusters, num_features=num_features)
    
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 运行遗传算法
    result_pop, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                              stats=stats, verbose=True)


    # 获取最优个体
    top_individuals = tools.selBest(result_pop, k=1)
    best_individual = top_individuals[0]
    print(best_individual)
    
    # 提取聚类中心
    best_centroids = np.array(best_individual).reshape(-1, data.shape[1])
    labels = np.argmin(np.linalg.norm(data[:, None] - best_centroids, axis=2), axis=1)

    plt.figure(figsize=(8, 6))
    for i in range(num_clusters):
        cluster_data = data[labels == i]
        plt.scatter(cluster_data[:, 2], cluster_data[:, 3], label=f'Cluster {i}')
    plt.scatter(best_centroids[:, 2], best_centroids[:, 3], s=300, c='red', marker='x', label='Centroids')
    plt.legend()
    plt.title('Iris Data Clustering using NSGA-II')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    # print(population[0:2])
    # return population
    
if __name__ == "__main__":
    main()