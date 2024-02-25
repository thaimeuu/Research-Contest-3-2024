# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:04:22 2022

@author: 佡儁

@editor: thai (add decision graph, handle pairwise distance as an input)
"""
"""
DPC算法，返回聚类的结果

DPC algorithm, returns clustering results
"""
import numpy as np
import math
import scipy.spatial.distance
import matplotlib.pyplot as plt


def DPC(data, k):
    """
    density peaks clustering

    Args:
        data (ndarray (m, n)): data points
        k (int): number of clusters

    Returns:
        cl: labels
    """

    ratio = 2

    # in case input is already a pairwise distance list
    if data.shape[1] == 3 and abs(np.max(np.unique(data[:, 0])).astype('i') - np.max(np.unique(data[:, 1])).astype('i')) == 1:
        if len(np.unique(data[:, 0])) > len(np.unique(data[:, 1])):
            Num = len(np.unique(data[:, 0]))
        else:
            Num = len(np.unique(data[:, 1]))
        distance = data[:, 2]
    else:
        Num = data.shape[0]
        # pairwise distance - pdist
        distance = scipy.spatial.distance.pdist(data)

    # distance matrix
    dis_matrix = scipy.spatial.distance.squareform(distance)
    sda = np.sort(distance)

    area = sda[round(sda.shape[0] * ratio / 100) - 1]  # calculate d_c
    # Find density
    density = np.zeros(Num, dtype=np.int32)
    for i in range(Num - 1):
        for j in range(i + 1, Num):
            if dis_matrix[i, j] < area:
                density[i] += 1
                density[j] += 1

    maxd = dis_matrix.max()

    # sort data points by density in descending order and return its index
    density_index = np.argsort(-density, kind="stable")
    delta = np.zeros(Num, dtype=np.float32)
    nneigh = np.zeros(Num, dtype=np.int32)

    # find delta
    delta[density_index[0]] = -1.0
    for i in range(1, Num):
        delta[density_index[i]] = maxd
        for j in range(i):
            if dis_matrix[density_index[i], density_index[j]] < delta[density_index[i]]:
                delta[density_index[i]] = dis_matrix[density_index[i], density_index[j]]
                nneigh[density_index[i]] = density_index[j]
    delta[density_index[0]] = delta.max()

    gamma = density * delta

    # print("\ndensity\n", density[:100], "\ndelta\n", delta[:100], "\ngamma\n", gamma[:100], "\nnneigh\n", nneigh[:100])
    # print("\ndensity_index\n", density_index[:100])
    gamma_index = np.argsort(-gamma, kind="stable")
    cluster = gamma_index[:k]  # The number of the cluster center
    cl = np.ones(Num, dtype=np.int32)
    cl *= -1
    icl = np.ones(cluster.shape[0] + 1, dtype=np.int32)
    icl *= -1
    # print("\ncl\n", cl[:100])
    Num_cluster = 0
    # print("cluster size: ", cluster.shape[0])
    for i in range(cluster.shape[0]):
        Num_cluster = Num_cluster + 1
        cl[gamma_index[i]] = Num_cluster
        icl[Num_cluster] = gamma_index[
            i
        ]  # 第Num_cluster个簇的簇心是i -> The cluster center of the Num_cluster cluster is i
    # print("\ncl\n", cl[:100])

    for i in range(Num):
        if cl[density_index[i]] == -1:
            cl[density_index[i]] = cl[nneigh[density_index[i]]]

    # plot decision graph
    plt.scatter(np.arange(1, Num + 1), np.sort(gamma)[::-1])
    plt.title("Decision graph")
    plt.ylabel("gamma = rho * delta")
    plt.show()

    return cl
