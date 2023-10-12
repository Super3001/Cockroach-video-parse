# kmeans.py

import random
import numpy as np
# from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 初始化簇心
def get_init_centers(raw_data, k):
    indices =  np.random.choice(raw_data.shape[0], k)
    return raw_data[indices]

# 计算距离
def cal_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

# 将各点分配到最近的点, 并计算MSE
def get_cluster_with_mse(raw_data, centers):
    distance_sum = 0.0
    cluster = {}
    for item in raw_data:
        flag = -1
        min_dis = float('inf')
        for i, center_point in enumerate(centers):
            dis = cal_distance(item, center_point)
            if dis < min_dis:
                flag = i
                min_dis = dis
        if flag not in cluster:
            cluster[flag] = []
        cluster[flag].append(item)
        distance_sum += min_dis**2
    return cluster, distance_sum/(len(raw_data)-len(centers))

# 计算各簇的中心点，获取新簇心
def get_new_centers(cluster):
    center_points = []
    for key in cluster.keys():
        center_points.append(np.mean(cluster[key], axis=0)) # axis=0，计算每个维度的平均值
    return center_points

# K means主方法
def k_means(raw_data, k, mse_limit, early_stopping):
    old_centers = get_init_centers(raw_data, k)
    old_cluster, old_mse = get_cluster_with_mse(raw_data, old_centers)
    new_mse = 0
    count = 0
    while np.abs(old_mse - new_mse) > mse_limit and count < early_stopping : 
        old_mse = new_mse
        new_center = get_new_centers(old_cluster)
        print(new_center)
        new_cluster, new_mse = get_cluster_with_mse(raw_data, new_center)  
        count += 1
        old_cluster = new_cluster
        print('mse:',np.abs(new_mse), 'Update times:',count)
    # print('cluster count', [len(x) for x in new_cluster.values()])
    # print('centers',new_center)
    return new_cluster, new_center

if __name__ == '__main__':
    #模拟数据集
    # X,y = make_blobs (n_samples = 100,
                    # cluster_std = [0.3,0.3,0.3],
                    # centers = [[0,0],[1,1],[-1,1]],
                    # random_state = 3)

    # print(type(X)) # np.ndarray
    
    X = []

    clusters, centers = k_means(X, 3, 0.01, 10)

    cls_1 = np.array(clusters[0]).T
    cls_2 = np.array(clusters[1]).T
    cls_3 = np.array(clusters[2]).T

    plt.scatter(cls_1[0], cls_1[1], c='r')
    plt.scatter(cls_2[0], cls_2[1], c='g')
    plt.scatter(cls_3[0], cls_3[1], c='b')
    
    centers = np.array(centers).T
    plt.scatter(centers[0], centers[1], c='y', marker='*')
    
    plt.show()
    