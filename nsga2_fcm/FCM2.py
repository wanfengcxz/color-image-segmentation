from pylab import *
import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score  # NMI
from sklearn.metrics import rand_score  # RI
from sklearn.metrics import accuracy_score  # ACC
from sklearn.metrics import f1_score  # F-measure
from sklearn.metrics import adjusted_rand_score  # ARI



# 初始化模糊矩阵U
def initializeMembershipMatrix(df, c):
    num_samples = df.shape[0]
    membership_mat = np.random.rand(num_samples, c)
    membership_mat = membership_mat / np.sum(membership_mat, axis=1, keepdims=True)
    return membership_mat


# 计算类中心点
def calculateClusterCenter(membership_mat, c, m, dataset):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    cluster_mem_val_list = list(cluster_mem_val)
    for j in range(c):
        x = cluster_mem_val_list[j]
        x_raised = [e ** m for e in x]
        denominator = sum(x_raised)
        temp_num = list()
        for i in range(membership_mat.shape[0]):
            data_point = list(dataset.iloc[i])
            prod = [x_raised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z / denominator for z in numerator]  # 每一维都要计算。
        cluster_centers.append(center)
    return cluster_centers


# 更新隶属度
def updateMembershipValue(membership_mat, cluster_centers, c, dataset):
    #    p = float(2/(m-1))
    data = []
    for i in range(membership_mat.shape[0]):
        x = list(dataset.iloc[i])  # 取出文件中的每一行数据
        data.append(x)
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(c)]
        for j in range(c):
            den = sum([math.pow(float(distances[j] / distances[k]), 2) for k in range(c)])
            membership_mat[i][j] = float(1 / den)
    return membership_mat, data


# 得到聚类结果
def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(membership_mat.shape[0]):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


# FCM算法
def fuzzyCMeansClustering(df, dataset, c, epsilon, m, T):
    start = time.time()  # 开始时间，计时
    membership_mat = initializeMembershipMatrix(df, c)  # 初始化隶属度矩阵
    print(membership_mat.shape)
    t = 0
    while t <= T:  # 最大迭代次数
        cluster_centers = calculateClusterCenter(membership_mat, c, m, dataset)  # 根据隶属度矩阵计算聚类中心
        old_membership_mat = membership_mat.copy()  # 保留之前的隶属度矩阵，同于判断迭代条件
        membership_mat, data = updateMembershipValue(membership_mat, cluster_centers, c, dataset)  # 新一轮迭代的隶属度矩阵
        cluster_labels = getClusters(membership_mat)  # 获取标签
        if np.linalg.norm(membership_mat - old_membership_mat) < epsilon:
            break
        print("第", t, "次迭代")
        t += 1

    print("用时：{0}".format(time.time() - start))
    # print(membership_mat)
    return cluster_labels, cluster_centers, data, membership_mat


# 计算聚类指标
def clustering_indicators(df, columns, labels_true, labels_pred):
    if type(labels_true[0]) != int:
        labels_true = LabelEncoder().fit_transform(df[columns[len(columns) - 1]])  # 如果标签为文本类型，把文本标签转换为数字标签
    f_measure = f1_score(labels_true, labels_pred, average='macro')  # F值
    accuracy = accuracy_score(labels_true, labels_pred)  # ACC
    normalized_mutual_information = normalized_mutual_info_score(labels_true, labels_pred)  # NMI
    rand_index = rand_score(labels_true, labels_pred)  # RI
    ARI = adjusted_rand_score(labels_true, labels_pred)
    return f_measure, accuracy, normalized_mutual_information, rand_index, ARI


def load_dataset():
    # 数据保存在.csv文件中
    df = pd.read_csv("dataset/iris_training.csv", header=0)  # 鸢尾花数据集 Iris  class=3
    # wine = pd.read_csv("dataset/wine.csv")  # 葡萄酒数据集 Wine  class=3
    # seeds = pd.read_csv("dataset/seeds.csv")  # 小麦种子数据集 seeds  class=3
    # wdbc = pd.read_csv("dataset/wdbc.csv")  # 威斯康星州乳腺癌数据集 Breast Cancer Wisconsin (Diagnostic)  class=2
    # glass = pd.read_csv("dataset/glass.csv")  # 玻璃辨识数据集 Glass Identification  class=6
    
    columns = list(df.columns)  # 获取数据集的第一行，第一行通常为特征名，所以先取出
    features = columns[:len(columns) - 1]  # 数据集的特征名（去除了最后一列，因为最后一列存放的是标签，不是数据）
    dataset = df[features]  # 预处理之后的数据，去除掉了第一行的数据（因为其为特征名，如果数据第一行不是特征名，可跳过这一步）
    original_labels = list(df[columns[-1]])  # 原始标签（最后一列）
    
    return df, dataset, original_labels, columns


def FCM(c = 3, T = 100, m = 2.0, epsilon = 1e-5):
    # 聚类簇数 最大迭代数 模糊参数
    df, dataset, original_labels, columns = load_dataset()
    labels, centers, data, membership = fuzzyCMeansClustering(df, dataset, c, epsilon, m, T)  # 运行FCM算法
    return np.array(membership), np.array(centers)

if __name__ == "__main__":
    
    membership, centers = FCM()
    print(membership.shape, centers.shape)

