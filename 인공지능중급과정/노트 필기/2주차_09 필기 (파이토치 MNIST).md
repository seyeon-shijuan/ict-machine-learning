
#### 모델 정의
- 파이토치에서 모델 정의를 위해서는 module을 상속한 클래스를 사용
- 계층(layer): 모듈 또는 모듈을 구성하는 한 개의 계층으로 합성곱층(convolutional layer), 선형 계층(linear layer) 등이 있음
- 모듈(module): 한 개 이상의 레이어가 모여 구성된 것
- 모델(model): 최종적으로 원하는 네트워크로, 한 개의 모듈이 모델이 될 수도 있음


>_MLP 란_ 여러 개의 퍼셉트론 뉴런을 여러 층으로 쌓은 다층신경망 구조. 입력층과 출력층 사이에 하나 이상의 은닉층을 가지고 있는 신경망이다.

##### nn.Module()을 상속하여 정의하는 방법
```python
class MLP(nn.Module):
    def __init__(self, inputs):
        super(MLP, self).__init__()
        self.layer = Linear(inputs, 1) # 계층 정의
        self.activation = Sigmoid() # 활성화 함수
        
    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X
```

- __init__()에서는 모델에서 사용할 모듈(nn.Linear, nn.Conv2d), 활성화 함수 등을 정의
- forward() 에서는 모델에서 실행해야 하는 연산 정의

nn.Model : 복잡한 구조
nn.Sequential : 간단한 구조

nn.Sequential을 사용하면 init에서 네트워크 모델을 정의해주고, forward에서도 연산을 순차적으로 실행되게하는데 가독성이 뛰어나게 작성 가능


#### MLP

```python
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
    
```

- convolutional layer: 이미지의 특징을 학습시키는것
- layer1, layer2에서 이미지의 특징을 학습함
- in_channels = 3은 RGB
- out_channels = 64은 convolutional 처리 후 출력되는 채널 수(filter의 개수)
- kernel_size = 5는 출력 필터의 사이즈로 정사각형 5x5 필터를 통해 특징을 추출
- ReLU()는 활성화 함수로 여기서 나온 값을 pooling층으로 넘김
- ReLU()의 inplace는 기존의 변수에 덮어쓰기할 지 여부


```python
model = MLP()

print(list(model.children()))
print(list(model.modules()))

# result

[Sequential(
  (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
), Sequential(
  (0): Conv2d(64, 30, kernel_size=(5, 5), stride=(1, 1))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
), Sequential(
  (0): Linear(in_features=750, out_features=10, bias=True)
  (1): ReLU(inplace=True)
)]

[MLP(
  (layer1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (layer2): Sequential(
    (0): Conv2d(64, 30, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (layer3): Sequential(
    (0): Linear(in_features=750, out_features=10, bias=True)
    (1): ReLU(inplace=True)
  )
), Sequential(
  (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
), Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1)), ReLU(inplace=True), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), Sequential(
  (0): Conv2d(64, 30, kernel_size=(5, 5), stride=(1, 1))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
), Conv2d(64, 30, kernel_size=(5, 5), stride=(1, 1)), ReLU(inplace=True), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), Sequential(
  (0): Linear(in_features=750, out_features=10, bias=True)
  (1): ReLU(inplace=True)
), Linear(in_features=750, out_features=10, bias=True), ReLU(inplace=True)]
```


#### 모델의 파라미터 정의

손실함수(loss function)
- wx + b를 계산한 값과 실제 값인 y의 오차를 구해서 모델의 정확성을 측정

옵티마이저(optimizer)
- 데이터와 손실 함수를 바탕으로 모델의 업데이트 방법을 결정
- step() 으로 weight과 bias(파라미터)를 업데이트
- 모델의 파라미터별로 다른 학습률을 적용할 수 있음
- torch.optim.Optimizer(params, defaults) 클래스 사용
- zero_grad(): 옵티마이저에 사용된 파라미터들의 기울기(gradient)를 0으로 만듦
- torch.optim.lr_scheduler는 epoch에 따라 학습률을 조정
- ex) optim.Adadelta, optim.Adagrad, optim.Adam, optim.SparseAdam, optim,Adamax
- optim.ASGD, optim.LBFGS
- optim.RMSProp, optim.Rprop, optim.SGD

#### 모델 훈련
loss.backward() 로 기울기를 자동 계산
배치 반복 시에 오차가 중첩적으로 쌓이게 되면 zero_grad()를 사용하여 미분 값을 0으로 초기화


#### 모델 평가

```python
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
```

- randint(high, size, dtype=None, layout)


#### car evaluation

> price(자동차 가격)
> maint(자동차 유지 비용)
> doors(자동차 문 개수)
> persons(수용 인원)
> lug_capacity(수하물 용량)
> safety(안전성)
> output(차 상태): 이 데이터는 unacc(허용 불가능한 수준) 및 acc(허용 가능한 수준),
> 양호(good) 및 매우 좋은(very good, vgood) 중 하나의 값을 갖음


