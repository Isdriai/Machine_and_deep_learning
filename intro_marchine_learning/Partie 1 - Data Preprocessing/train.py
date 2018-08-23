#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 13:48:22 2018
https://www.udemy.com/introduction-au-machine-learning/learn/v4/

@author: picot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#################################### Preprocessing ####################################

dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values= 'NaN', strategy='mean', axis=0)

imputer.fit(x[:,1:3])

x[:, 1:3] = imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncode_country = LabelEncoder()
x[:,0] = labelEncode_country.fit_transform(x[:,0])

oneHotEncoder = OneHotEncoder(categorical_features=[0])

x = oneHotEncoder.fit_transform(x).toarray()


labelEncode_y = LabelEncoder()
y = labelEncode_y.fit_transform(y)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


