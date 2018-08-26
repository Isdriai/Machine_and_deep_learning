#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:28:14 2018

@author: picot
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#################################### Preprocessing ####################################

dataset = pd.read_csv("Position_Salaries.csv")


x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values


#################################### Construction du modele ####################################

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

regressor = LinearRegression()

regressor.fit(x_poly, y)

plt.scatter(x, y, color= 'red')
plt.plot(x, regressor.predict(x_poly), color ='blue')
plt.title('Salaire vs expérience')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.show()
