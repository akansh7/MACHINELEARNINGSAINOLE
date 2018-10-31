#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 19:08:23 2018

@author: akansh_7
"""
#importing the dataset
import pandas as pd
data=pd.read_csv("apple_sales.csv")

color=[]
for i in data['color']:
    if i=='red':
        color.append(1)
    else:
        color.append(0)
data['color']=color


#splitting the dataset into dependent and independent variables.
x = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

#fitting the decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)


#predicting the result  
regressor.predict([[0],[3]])
