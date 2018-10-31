#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:44:53 2018

@author: akansh_7
"""
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("CCdetails.csv")

#Removing the NaN values from CREDIT_LIMIT column
data['CREDIT_LIMIT']=data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean())

#selecting the columns for clustering
x = data.iloc[:,[13,14]].values

#finding optimal number of clusters using dendrograms
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('DENDROGRAM')
plt.xlabel('FEATURES')
plt.show()

#fitting the hierarchical clustering to the datset
from sklearn.cluster import AgglomerativeClustering
c = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
cluster = c.fit_predict(x)

#visualising
plt.scatter(x[cluster == 0,0], x[cluster == 0,1], s=10, c='blue', label = "C1")
plt.scatter(x[cluster == 1,0], x[cluster == 1,1], s=10, c='green', label = "C2")
plt.scatter(x[cluster == 2,0], x[cluster == 2,1], s=10, c='red', label = "C3")
plt.scatter(x[cluster == 3,0], x[cluster == 3,1], s=10, c='grey', label = "C4")
plt.scatter(x[cluster == 4,0], x[cluster == 4,1], s=10, c='pink', label = "C5")
plt.title("Cluster")
plt.xlabel("CREDIT_LIMIT")
plt.ylabel("PAYMENTS")
plt.legend()
plt.show()