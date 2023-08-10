# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:24:42 2023

@author: 예비
"""

import torch

# CUDA 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CPU 텐서 생성
cpu_tensor = torch.tensor([1, 2, 3])

# GPU 텐서 생성
gpu_tensor = torch.tensor([1, 2, 3]).to(device)

print("CPU Tensor:", cpu_tensor)
print("GPU Tensor:", gpu_tensor)


my_tensor = torch.tensor([[1., -1.], [1., -1.]])
