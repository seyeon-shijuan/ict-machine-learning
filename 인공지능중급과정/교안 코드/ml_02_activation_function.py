# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:23:21 2023

@author: SEYEON
"""

import numpy as np
import matplotlib.pyplot as plt

# 계단 함수
def step_function(x):
    return np.array(x>0, dtype=np.int_)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 시그모이드
def sigmoid(x):
    return 1 / (1 +np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 시그모이드 미분
def diff_sigmoid(x):
    return 1/(1+np.exp(-x)) * (1-(1/(1 + np.exp(-x))))


x = np.arange(-10.0, 10.0, 0.1)
y = diff_sigmoid(x)
#fig = plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.ylim(-0.5, 1.5)
plt.show()


# 탄젠트
x = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성
y = np.tanh(x)
plt.plot(x, y)
plt.plot([0,0],[1.0,-1.0], ':')
plt.axhline(y=0, color='orange', linestyle='--')
plt.title('Tanh Function')
plt.show()

# 렐루
def relu(x):
    return np.maximum(0, x)
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Relu Function')
plt.show()

# 리키 렐루
def leaky_relu(x, a=0.01):
    return np.maximum(a*x, x)

x = np.arange(-5.0, 5.0, 0.1)
y = leaky_relu(x)
plt.plot(x, y)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Leaky ReLU Function')
plt.show()


# 소프트맥스
def softmax(a):
    c = np.max(a) # overflow 방지를위한 변수
    exp_a = np.exp(a - c) # overflow 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))
      
      
