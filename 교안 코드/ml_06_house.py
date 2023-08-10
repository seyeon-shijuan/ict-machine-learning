# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:36:42 2023

@author: 예비
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_csv('./data/house_train.csv')

d_types = df.dtypes

# print(df.isnull().sum().sort_values(ascending=False).head(20))

df = pd.get_dummies(df)
df = df.fillna(df.mean())

# 속성별 관련도 추출
df_corr = df.corr()
df_corr_sort = df_corr.sort_values('SalePrice', ascending=False)
# print(df_corr_sort['SalePrice'].head())

# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
# sns.pairplot(df[cols])
# plt.show()

# cols = ['SalePrice', 'HalfBath', 'LotArea']
# sns.pairplot(df[cols])
# plt.show()


cols_train = ['OverallQual', 'GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
X_train_pre = df[cols_train]
y = df['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)


model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')


early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

modelpath = './data/model/Ch15-house.hdf5'
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32, callbacks=[early_stopping_callback, checkpointer])


real_prices = []
pred_prices = []
X_num = []

n_iter = 0

tmp = model.predict(X_test)
# 2차원 (292, 1)

Y_prediction = model.predict(X_test).flatten()
# 1차원 (292,)

for i in range(25):
    real = y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.2f}, 예상가격: {:.2f}".format(real, prediction))
    real_prices.append(real)

    pred_prices.append(prediction)
    n_iter = n_iter + 1
    X_num.append(n_iter)


plt.plot(X_num, pred_prices, label='predicted price')
plt.plot(X_num, real_prices, label='real price')
plt.legend()
plt.show()

