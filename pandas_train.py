# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import pandas as pd
import numpy as np

print(pd.__version__)

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })

print(str(cities) + "\n\n")


california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
print(california_housing_dataframe.describe())

print("\n\n\n")

california_housing_dataframe.hist('housing_median_age')

print(np.log(population))

print("\n\n\n")

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']


cities['Grande ville sainte'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: "San" in name)

print("\n\n\n")
print("\n\n\n")
print(cities)

print("\n\n\n")
print("\n\n\n")

print(cities.reindex(np.random.permutation(cities.index)))