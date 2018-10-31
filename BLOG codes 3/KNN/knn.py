#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 01:42:56 2018

@author: akansh_7
"""
#importing the dataset
import pandas as pd
data=pd.read_csv("diabetes.csv")
    
#removing unnecessary columns
data.drop({'Insulin', 'SkinThickness'}, axis=1, inplace=True)

#splitting the dataset into dependent and independent variales
x = data.iloc[:, 0:6].values
y = data.iloc[:, 6].values

#splitting the test data and train data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 0)

#fitting K-Nearest Neighbor to our training data
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
classifier.fit(x_train, y_train)

#predicting the result
prediction = classifier.predict(x_test)

#cheking the total correct and incorrect predictions
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, prediction)