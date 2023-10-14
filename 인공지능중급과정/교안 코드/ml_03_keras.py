# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:42:44 2023

@author: 예비
"""
# import tensorflow

# 딥러닝 과정
# 1. 데이터 수집
# 2. 데이터 전처리
# 3. 모델 학습(3-1. 모델 생성, 3-2. 층 구성, 3-3 모델 환경설정 3-4 모델 학습)
# 4. 모델 검증 및 평가(검증데이터 및 테스트 데이터 활용)
# 5. 모델 예측(실제 데이터를 활용한 예측)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data_set = np.loadtxt('./data/ThoraricSurgery3.csv', delimiter=',')
# data_set1 = pd.load_csv("./data/ThoraricSurgery3.csv", delimiter=',')

x = data_set[:, 0:16]
y = data_set[:, 16]
             
# 모델 생성 및 층 구성
model = Sequential()
# 입력, 은닉층
model.add(Dense(30, input_dim=16, activation='relu'))
# 출력층
model.add(Dense(1, activation='sigmoid'))
# 모델 환경설정(손실함수, 옵티마이저, 평가지표)
model.compile(loss='binary_crossentropy', optimizer='adam')

# X 입력 y 정답데이터
history = model.fit(x, y, epochs=5, batch_size=16)

