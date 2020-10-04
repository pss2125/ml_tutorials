#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 07:24:13 2020

@author: siva
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("mall.csv")
X = dataset.iloc[:,[3,4]].values

#use dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.title("Dendogram")
plt.xlabel("customers")
plt.ylabel("Euclidean Distances")
left, right = plt.xlim()
plt.xticks(np.arange(start=left, stop= right, step=500))
plt.show()

#hierarchial clustering using agglomerative
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage="ward")
y_hc = hc.fit_predict(X)

#visualizing cluster
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], c="cyan", label="cluster1")
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], c="salmon", label="cluster2")
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], c="grey", label="cluster3")
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], c="green", label="cluster4")
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], c="magenta", label="cluster5")
plt.title("Hierarchial clustering")
plt.xlabel("Annual Income k$")
plt.ylabel("Spending Score 1-100")
plt.legend()
plt.show()
