# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:29:48 2023

@author: 예비
"""

import numpy as np


a = np.array([1,2,3])
print(a)
print(a[0])

b = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(b)
print(b[0][2])

a = np.array([[0,1,2],[3,4,5],[6,7,8]])
print(a.shape) # 배열의 형상
print(a.ndim) # 배열의 차원 개수
print(a.dtype) # 요소의 자료형
print(a.itemsize) # 요소 한개의 크기
print(a.size) # 전체 요소의 개수

print(np.arange(5))
print(np.arange(1,6))

print("---------")
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
tmp2 = np.concatenate((x, y), axis=1)
print(tmp2)
tmp = np.vstack((x, y))
print(tmp)

vxy = np.hstack

tmp_a = [1,2,3]
tmp_b = [4,5,6]
tmp_ab = np.dot(a,b)
print(tmp_ab)