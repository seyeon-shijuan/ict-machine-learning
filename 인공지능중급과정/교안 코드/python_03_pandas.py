# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:01:03 2023

@author: 예비
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# !git clone https://github.com/taehojo/data.git

df = pd.read_csv('data/pima-indians-diabetes3.csv')

head_df = df.head(5)
val_df = df["diabetes"].value_counts()
# 0: 당뇨x 1: 당뇨o
print(df.describe())
corr_df = df.corr()

colormap = plt.cm.gist_heat # 그래프 색상 구성
plt.figure(figsize=(12,12)) # 그래프 크기

sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap,
            linecolor='white', annot=True)
plt.show()


plt.hist(x=[df.plasma[df.diabetes==0], df.plasma[df.diabetes==1]],
         bins=30, histtype='barstacked', label=['normal','diabetes'])
plt.legend()
plt.show()


plt.hist(x=[df.bmi[df.diabetes==0], df.bmi[df.diabetes==1]],
         bins=30, histtype='barstacked', label=['normal', 'diabetes'])
plt.legend()
plt.show()


X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

# 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu',
                name='Dense_1'))
model.add(Dense(8, activation='relu',
                name='Dense_2'))
model.add(Dense(1, activation='sigmoid', 
                name='Dense_3'))
model.summary()

# 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# 모델 실행
# history = model.fit(X, y, epochs=100, batch_size=5)


# plt.hist(x=[df.iloc[:, 6][df.iloc[:, 16]==0], 
#             df.iloc[:, 6][df.iloc[:, 16]==1]],
#          bins=30, histtype='barstacked', 
#          label=['normal', 'diabetes'])
# plt.show()

###########################################

'''
fixed acidity : 고정 산도
volatile acidity : 휘발성 산도
citric acid : 시트르산
residual sugar : 잔류 당분
chlorides : 염화물
free sulfur dioxide : 자유 이산화황
total sulfur dioxide : 총 이산화황
density : 밀도
pH
sulphates : 황산염
alcohol
quality : 0 ~ 10(높을 수록 좋은 품질)
class :  와인종류(1:레드와인, 0:화이트와인)
'''
#와인 데이터 분석
df = pd.read_csv('data/wine.csv', delimiter=',', header=None)
#print(df.iloc[:, 12].value_counts())

x = df.iloc[:, 0:12]
y = df.iloc[:, 12]

colormap = plt.cm.gist_heat
plt.figure(figsize=(12, 12))

# sns.heatmap(df.corr(), linewidths=0.1, vmax=0.6, cmap=colormap, linecolor='white', annot=True)
# plt.show()

plt.hist(x=[df.iloc[:, 1][df.iloc[:,12]==0], df.iloc[:, 1][df.iloc[:,12]==1]], bins=10, histtype='barstacked', label=['whitewine', 'redwine'])
# plt.hist(x=[df.iloc[:, 4][df.iloc[:,12]==0], df.iloc[:, 4][df.iloc[:,12]==1]], bins=5, histtype='barstacked', label=['whitewine', 'redwine'])
plt.legend()


###########################################
df = pd.read_csv('data/iris3.csv')
sns.pairplot(df, hue='species')
plt.figure(figsize=(12, 12))
plt.show()


###########################################
# keras 이용한 Iris 품종 예측

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# 원 핫 인코딩
y = pd.get_dummies(y)


# 모델 설정
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])


# 층 4개, 은닉층 2개
# 입력 데이터 4개

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])


# 모델 실행
history = model.fit(X, y, epochs=50, batch_size=5)


