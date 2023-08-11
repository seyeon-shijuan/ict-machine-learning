# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:46:31 2023

@author: 예비
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('./data/car_evaluation.csv')
print(dataset.shape)
print(dataset['output'].value_counts())


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
dataset.output.value_counts().plot(kind='pie', autopct='%0.05f%%',
                                   colors=['lightblue','lightgreen','orange','pink'],
                                   explode = (0.05, 0.05, 0.05, 0.05)
                                   )
plt.show()



dataset['output'].value_counts().plot(kind='bar', color=['lightblue', 'lightgreen', 'orange', 'pink'], legend=False)
plt.title("bar")
plt.show()


categorical_columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety']
categorical_columns = list(dataset.columns)[:-1]

for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')
    
tmp = dataset['price'].cat.codes

price = dataset['price'].cat.codes.values
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

categorical_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)
print(categorical_data[:10])



