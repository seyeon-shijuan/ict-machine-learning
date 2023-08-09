# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:25:49 2023

@author: 예비
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/wine.csv', header=None)
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]

# 학습셋 테스트셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델 구조 설정
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

# 모델 저장 조건 설정
# modelpath = "./data/model/all/{epoch:02d}-{val_accuracy:.4f}.hdf5"
modelpath = "./data/model/all/Ch14-4-bestmodel.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, verbose=0, monitor='val_loss', save_best_only=True)

# 모델 실행
history = model.fit(X_train, y_train, epochs=1000, batch_size=500, validation_split=0.25, verbose=0, callbacks=[early_stopping_callback, checkpointer])
# 0.8 x 0.25 = 0.2

# 테스트 결과 출력
score = model.evaluate(X_test, y_test)
print('Test accuracy: ', score[1])

# model.evaluate return 값은 [loss, accuracy]

hist_df = pd.DataFrame(history.history)
print(hist_df)

y_v_loss = hist_df['val_loss']
y_loss = hist_df['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_v_loss, "o-", c="red", markersize=2, label='Testset_loss')
plt.plot(x_len, y_loss, "o-", c="blue", markersize=2, label='trainset_loss')

plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)


l_model = load_model('./data/model/all/Ch14-4-bestmodel.hdf5')
l_model.summary()

l_score = l_model.evaluate(X_test, y_test)
print('Test accuracy :', l_score[1])