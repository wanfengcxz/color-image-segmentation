import cv2
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler

import numba
from numba import njit


def mean_shift(data):
    # 创建MeanShift对象
    meanshift = MeanShift()
    # 拟合数据
    meanshift.fit(data.reshape(-1, bands))

    return meanshift.labels_, meanshift.cluster_centers_


if __name__ == "__main__":
    # hyperspectral_image.shape = (height, width, bands)
    hsi_img = loadmat("..\\training_images\\pavia\\PaviaU.mat")["paviaU"][..., 31:32]
    height, width, bands = hsi_img.shape

    hsi_img = hsi_img / hsi_img.max()
    # cv2.imshow("img", data)
    # cv2.waitKey(0)

    # 获取聚类标签
    labels, centers = mean_shift(hsi_img)

    # 将标签转换回图像形状
    clustered_image = labels.reshape(height, width)

    plt.imshow(clustered_image, cmap="viridis")
    plt.title("MeanShift Clustering of Hyperspectral Image")
    plt.axis("off")
    plt.show()
