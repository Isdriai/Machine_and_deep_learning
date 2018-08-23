#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:13:20 2018

@author: picot
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#################################### Preprocessing ####################################

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

'''
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values= 'NaN', strategy='mean', axis=0)

imputer.fit(x[:,1:3])

x[:, 1:3] = imputer.transform(x[:,1:3])

'''

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncode_country = LabelEncoder()
x[:,3] = labelEncode_country.fit_transform(x[:,3])

oneHotEncoder = OneHotEncoder(categorical_features=[3])

x = oneHotEncoder.fit_transform(x).toarray()

x = x[:, 1:] 

'''
la variable dependante est numérique
labelEncode_y = LabelEncoder()
y = labelEncode_y.fit_transform(y)
'''

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


