import numpy as np
from numpy import loadtxt

init, max_itr = 0, 100  # initial, maximum iteration
cluster_size = 5  # number of cluster
dataset = loadtxt('sample_data.csv', delimiter=',')  # read data


# distance calculation
def distance(c, d):
    return np.sqrt(((d[0] - c[0]) ** 2) + ((d[1] - c[1]) ** 2))  # distance


# distance distance of data with each centroid
def distance_calc(centd, data):
    D0 = []
    for i in range(len(centd)):
        tem = []
        for j in range(len(data)):
            tem.append(distance(centd[i], data[j]))  # distance of data with ith centroid
        D0.append(tem)
    return D0


# centroid update
def update_centroid(dt):
    return np.mean(np.array(dt).T, axis=1)  # mean of datas in same cluster


# Main
centroid = []
while init < max_itr:
    if init == 0:
        for i in range(cluster_size):
            centroid.append(dataset[i])  # initial centroid

    dist = distance_calc(centroid, dataset)  # distance
    clustered_group = np.argmin(np.array(dist).T, axis=1)  # clustered group

    # centroid update
    for nth_cluster in range(cluster_size):  # each cluster
        clustered_data = []
        for ii in range(len(clustered_group)):
            if nth_cluster == clustered_group[ii]:
                clustered_data.append(dataset[ii])  # get datas in each cluster

        # if more than 1 data in cluster, update centroid
        if len(clustered_data) > 1:
            centroid[nth_cluster] = update_centroid(clustered_data)

    init += 1

print("\nFinal Cluster: ", clustered_group)
