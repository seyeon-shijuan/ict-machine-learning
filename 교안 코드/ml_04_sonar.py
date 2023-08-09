# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:14:19 2023

@author: 예비
"""
# 광물 데이터

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

df = pd.read_csv('./data/sonar3.csv', header=None)

print(df.head())

print(df[60].value_counts())

# 변수, 결괏값 분리
X = df.iloc[:, 0:60]
y = df.iloc[:, 60]

# 학습 셋과 테스트 셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# 모델 설정
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
# model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델을 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# 모델 실행
history = model.fit(X_train, y_train, epochs=200, batch_size=10)

score = model.evaluate(X_test, y_test)
print('Test accuracy: ', score[1])

# 모델 저장
model.save('./data/model/my_model.h5')

del model

# 모델 불러오기
model = load_model('./data/model/my_model.h5')
score = model.evaluate(X_test, y_test)
print('Loaded Test accuracy: ', score[1])

del model

# KFold
k = 5
kfold = KFold(n_splits=k, shuffle=True)
acc_score = []
def model_fn():
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = model_fn()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0)
    
    accuracy = model.evaluate(X_test, y_test)[1]
    acc_score.append(accuracy)
    

# k 번 실시된 정확도의 평균을 구합니다.
avg_acc_score = sum(acc_score) / k

# 결과 출력

print('정확도: ', acc_score)
print('정확도 평균', avg_acc_score)
    




