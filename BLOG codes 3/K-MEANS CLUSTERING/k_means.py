#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:44:53 2018

@author: akansh_7
"""
import pandas as pd
data = pd.read_csv("CCdetails.csv")

#Removing the NaN values from CREDIT_LIMIT column
data['CREDIT_LIMIT']=data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean())

#selecting the columns for clustering
x = data.iloc[:,[13,14]].values

#finding the number of clusters using the elbow method
from sklearn.cluster import KMeans
a=[] # stores the mean of every cluster
for i in range(1, 11):
    kmeans=KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    a.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), a,color="pink")
plt.title("The Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("a")
plt.show()

#fitting the K-means algorithm to our dataset
kmeans=KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
cluster = kmeans.fit_predict(x)

#visualising
plt.scatter(x[cluster == 0,0], x[cluster == 0,1], s=10, c='blue', label = "C1")
plt.scatter(x[cluster == 1,0], x[cluster == 1,1], s=10, c='green', label = "C2")
plt.scatter(x[cluster == 2,0], x[cluster == 2,1], s=10, c='red', label = "C3")
plt.scatter(x[cluster == 3,0], x[cluster == 3,1], s=10, c='grey', label = "C4")
plt.scatter(x[cluster == 4,0], x[cluster == 4,1], s=10, c='pink', label = "C5")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=30, c="yellow")
plt.title("Cluster")
plt.xlabel("CREDIT_LIMIT")
plt.ylabel("PAYMENTS")
plt.legend()
plt.show()

