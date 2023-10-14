# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 15:23:41 2023

@author: Administrator
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 가상의 데이터셋과 모델 생성 (예시)
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(100, 10)  # 예시 데이터
        self.targets = torch.randint(2, (100,))  # 예시 타겟

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 모델 정의 (예시)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # 입력 차원 10, 출력 차원 1

    def forward(self, x):
        return self.fc(x)

# 하이퍼파라미터 설정
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로더 준비 (예시)
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 모델 및 손실 함수, 최적화 알고리즘 설정 (예시)
model = MyModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# TensorBoard SummaryWriter 설정
writer = SummaryWriter('./summary/tensorboard')

# 훈련 루프
for epoch in range(num_epochs):
    model.train()
    batch_loss = 0.0
    
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device).float(), y.to(device).float()
        outputs = model(x)
        loss = criterion(outputs, y)
        writer.add_scalar("LOSS", loss, epoch * len(dataloader) + i)  # 배치마다 손실 값 기록
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# SummaryWriter 닫기
writer.close()