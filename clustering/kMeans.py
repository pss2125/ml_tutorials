#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 13:58:34 2020

@author: siva
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("mall.csv")
X = dataset.iloc[:,[3,4]].values

#elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_init=10 ,init="k-means++", n_clusters=i, max_iter=300, random_state = 0)
    #kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10 , random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title("elbow method")
plt.xlabel("no of clusters")
plt.xticks(range(1,11))
plt.yticks(np.arange(start=0, stop=max(wcss)+30000, step=25000))
plt.ylabel("wcss")
plt.show()

#applying kmeans after elbow method
kmeans = KMeans(n_init=10, init="k-means++", n_clusters=5, max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(X)

#visualizing clusters
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, c ="wheat", label="Cluster 1")
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, c = "salmon", label="Cluster 2")
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, c = "plum", label="Cluster 3")
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, c = "yellow", label="Cluster 4")
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100, c = "cyan", label="Cluster 5")

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c="blue")
plt.title("Client Clusters")
plt.xlabel("Annual Income")
plt.ylabel("spending score")
plt.legend()
plt.show()






