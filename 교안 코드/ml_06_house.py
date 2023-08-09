# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:36:42 2023

@author: 예비
"""

import pandas as pd

df = pd.read_csv('./data/house_train.csv')

d_types = df.dtypes

print(df.isnull().sum().sort_values(ascending=False).head(20))