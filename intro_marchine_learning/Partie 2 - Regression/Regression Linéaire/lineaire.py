#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 17:20:39 2018

@author: picot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#################################### Preprocessing ####################################

dataset = pd.read_csv("Salary_Data.csv")


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1.0/3, random_state = 0)

#################################### Construction du modele ####################################

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)


y_pred = regressor.predict(x_test)

print(regressor.predict(15))


plt.scatter(x_test, y_test, color= 'red')
plt.plot(x_train, regressor.predict(x_train), color ='blue')
plt.title('Salaire vs expérience')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.show()
