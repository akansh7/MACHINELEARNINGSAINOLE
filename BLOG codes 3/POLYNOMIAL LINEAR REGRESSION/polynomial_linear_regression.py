#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:38:36 2018

@author: akansh_7
"""

#Importing the dataset
import pandas as pd
data=pd.read_csv("school.csv")

#separating the independent and dependent variables
x=data.iloc[:,1:2] # selects the second column data
y=data.iloc[:,2]

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 5)
polynomial_x=polynomial_regression.fit_transform(x)

from sklearn.linear_model import LinearRegression
#creating a LinearRegression object
linear_reg = LinearRegression()
#fitting the linear regression object on the polynomial matrix of features and the dependent variables
linear_reg.fit(polynomial_x, y)
#Polynomial regression model got created

#Viewing our model 
import matplotlib.pyplot as plt
plt.scatter(x, y, color='green')
plt.plot(x, linear_reg.predict(polynomial_regression.fit_transform(x)), color = 'blue')
plt.title('POLYNOMIAL REGRESSION')
plt.xlabel('NEED')
plt.ylabel('POINTS')

#predicting results 
linear_reg.predict(polynomial_regression.fit_transform(3.5))
linear_reg.predict(polynomial_regression.fit_transform(6.5))
