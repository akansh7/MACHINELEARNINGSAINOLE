#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 19:32:01 2018

@author: akansh_7
"""
#importing the dataset
import pandas as pd
data=pd.read_csv("insurance.csv")

#removing the region column
data.drop({'region'},axis=1,inplace=True)

#Dealing with the categorical data
#creating two empty lists where we will store the changed values.
sex=[]
smoker=[]
#checking the columns and appending values to the empty lists accordingly.
for i in data["sex"]:
    if i=='female':
        sex.append(1)
    else:
        sex.append(0)
for i in data["smoker"]:
    if i=='yes':
        smoker.append(1)
    else:
        smoker.append(0)
#Finally replacing the categorical values with the numeric values
data['sex']=sex
data['smoker']=smoker


#Separating the independent and dependent variables.
x=data.iloc[:,:-1].values
y=data.iloc[:,5].values

#splitting the dataset into Training and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)

#applying random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)

#predicting the results
prediction = regressor.predict(x_test)
