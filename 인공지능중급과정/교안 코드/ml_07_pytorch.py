# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:24:42 2023

@author: 예비
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# CUDA 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CPU 텐서 생성
cpu_tensor = torch.tensor([1, 2, 3])

# GPU 텐서 생성
gpu_tensor = torch.tensor([1, 2, 3]).to(device)

print("CPU Tensor:", cpu_tensor)
print("GPU Tensor:", gpu_tensor)


my_tensor = torch.tensor([[1., -1.], [1., -1.]])

# tensor를 numpy의 ndarray로 변환하기
temp = torch.tensor([[1, 2], [3, 4]])
tmp_np = temp.numpy()

# GPU로 텐서 만들기
# temp = torch.tensor([1, 2], [3, 4], device="cuda:0")
# print(temp.to("cpu").numpy())


# tensor 인덱스 접근
temp = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7])
print(temp[0], temp[1], temp[2])
print('-------------')
print(temp[2:5], temp[4:-1])

# tensor 차원 조작
temp = torch.tensor([[1, 2], [3, 4]]) # 2x2 행렬

print(temp.shape)
print(temp.view(4, 1)) # 4x1로 변경
print(temp.view(-1)) # 1차원 벡터로 변경
print(temp.view(1, -1)) # -1은 (1,?)의 ?를 자동 계산해서 처리
# 결론은 1x4로 바뀌게 됨
print(temp.view(-1, 1))

print('----------------')
data = pd.read_csv('data/class2.csv')

x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()
y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()


class CustomDataset(Dataset):
    # 필요한 변수를 선언하고, 데이터셋의 전처리를 하는 함수
    def __init__(self, csv_file):
        self.label = pd.read_csv(csv_file)
    
    # 데이터셋의 길이, 즉 총 샘플의 수를 가져오는 함수
    def __len__(self):
        return len(self.label)
    
    # 데이터셋에서 특정 데이터를 가져오는 함수(index 번째 데이터를 반환하는 함수이며, 이때 반환되는 값은 텐서 형태)
    def __getitem__(self, idx):
        sample = torch.tensor(self.label.iloc[idx, 0:3]).int()
        label = torch.tensor(self.label.iloc[idx, 3]).int()
        return sample, label
    

tensor_dataset = CustomDataset('data/covtype.csv')
dataset = DataLoader(tensor_dataset, batch_size=4, shuffle=True)


for i, data in enumerate(dataset, 0):
    print(i, end=' ')
    batch = data[0]
    print(batch.size())
    

