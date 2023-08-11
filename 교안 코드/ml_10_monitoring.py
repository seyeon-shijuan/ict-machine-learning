# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:36:41 2023

@author: 예비
"""
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.nn as nn
import requests
import torch
import torchmetrics
import torch.utils.tensorboard import SummaryWriter


model.eval()
with torch.no_grad():
    valid_loss = 0

    for x, y in valid_dataloader:
        outpus = model(x)
        loss = F.cross_entropy(outputs, y.long().squeeze()) # 차원축소
        valid_loss += float(loss)
        y_hat += [outputs]
        
valid_loss = valid_loss / len(valid_loader)

