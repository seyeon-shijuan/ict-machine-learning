# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:04:22 2023

@author: 예비
"""

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.nn as nn
import requests
import torch
import torchmetrics
import torch.utils.tensorboard import SummaryWriter


mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
    ]) # 평균이 0.5, 표준편차가 1.0이 되도록 데이터의 분포(normalize)를 조정


download_root = './data/MNIST_DATASET'

train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)

"""
class MLP(nn.Module):
    def __init__(self, inputs):
        super(MLP, self).__init__()
        self.layer = Linear(inputs, 1) # 계층 정의
        self.activation = Sigmoid() # 활성화 함수
        
    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X
    
"""

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # 선형 함수
        self.layer1 = nn.Sequential(
            # 특징 값 찾을 때 convolutional layer 사용
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            # pooling 층
            nn.MaxPool2d(2)
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=30*5*5, out_features=10, bias=True),
            nn.ReLU(inplace=True)
            )
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.layer3(x)
        return x
    

model = MLP()

print(list(model.children()))
print(list(model.modules()))


# 모델 파라미터 정의

# from torch.optim import optimizer

# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# scheduler = torch.optim.lr_scheduler(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

# for epoch in range(1, 100+1):
#     for x, y in dataloader:
#         optimizer.zero_grad()
#     loss_fn(model(x), y).backward()
#     optimizer.step()
#     scheduler.step()



# 모델 평가
metric = torchmetrics.Accuracy(task='multiclass', num_classes=5)

n_batches = 10
for i in range(n_batches):
    preds = torch.randn(10, 5).softmax(dim=-1)
    print(f"preds: {preds}")
    target = torch.randint(5, (10,))
    print(f"target: {target}")
    
    acc = metric(preds, target)
    print(f"Accuracy on batch {i}: {acc}")

acc = metric.compute()
print(f"Accuracy on all data: {acc}")


# 훈련과정 모니터링
writer = SummaryWriter("tensorboard")

for epoch in range(num_epochs):
    model.train() # 학습 모드로 전환
    batch_loss = 0.0
    
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device).float(), y.to(device).float()
        outputs = model(x)
        loss = criterion(outputs, y)
        writer.add_scalar("loss", loss, epoch) # 스칼라 값(오차)를 기록
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

writer.close()

    
