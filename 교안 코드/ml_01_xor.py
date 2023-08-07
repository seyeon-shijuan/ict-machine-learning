# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:11:16 2023

@author: 예비
"""

import numpy as np

# MLP로 XOR 문제 해결하기

w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1= 3
b2 = -1
b3 = -1

def mlp(x, w, b):
    y = np.sum(w*x) +b
    if y <= 0:
        return 0
    else:
        return 1

# nand 게이트
def NAND(x1, x2):
    return mlp(np.array([x1, x2]), w11, b1)

# or gate
def OR(x1,x2):
    return mlp(np.array([x1, x2]), w12, b2)


# and gate
def AND(x1, x2):
    return mlp(np.array([x1,x2]), w2, b3)

# xor gate
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))


for x in [(0,0), (1,0), (0,1), (1,1)]:
    y = XOR(x[0], x[1])
    print("입력 값: "+ str(x) + "출력 값: "+ str(y))
    

##############################
# 
##############################
    