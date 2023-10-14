### VGGNet

카렌 시모니안(Karen Simonyan)과 앤드류 지서만(Andrew Zisserman)이 2015
ICLR에 게재한 “Very deep convolutional networks for large-scale image recognition” 논문
에서 처음 발표
합성곱층의 파라미터 수를 줄이고 훈련 시간을 개선

![](https://blog.kakaocdn.net/dn/K990l/btqwDJ7C54R/664Ksm6gyTGBR1wK3YPDFk/img.png)

- 커널의 크기 3x3, 최대 풀링 커널 크기 2x2 스트라이드 2

### copy
얕은복사 vs 깊은 복사
얕은 복사는 복사 전 값을 바꿔도 복사한 것이 바뀌지 않음
얕은 복사는 copy.copy()를 이용

#### Code


```python
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
```
- eq는 equal의 약자로 서로 같은지를 비교하는 표현식
- view_as(other)는 other의 텐서 크기를 사용하겠다는 의미


```python
def train(model, iterator, optimizer, criterion, device):    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()    
    for (x, y) in iterator:        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()                
        y_pred, _ = model(x)        
        loss = criterion(y_pred, y)       
        acc = calculate_accuracy(y_pred, y)        
        loss.backward()        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```
- epoch_loss, epoch_acc 누적


#### 평가함수
```python
def evaluate(model, iterator, criterion, device):    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()    
    with torch.no_grad():        
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```
- torch.no_grad()를 사용하여 반복


#### 시간 측정 함수
```python
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```
- 시간 측정 함수



- time.monotonic() : 하드웨어에서 계산하는 시간

#### 모델 학습
```python
import time

EPOCHS = 5
best_valid_loss = float('inf')
for epoch in range(EPOCHS):    
    start_time = time.monotonic()    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'VGG-model.pt')

    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Valid. Loss: {valid_loss:.3f} |  Valid. Acc: {valid_acc*100:.2f}%')
```


#### 예측

```python
def get_predictions(model, iterator):
    model.eval()
    images = []
    labels = []
    probs = []
    
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_pred, _ = model(x)
            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)
            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)
    return images, labels, probs
```


- y_prob = F.softmax(y_pred, dim = -1) = -1은 맨 마지막 인덱스 의미
- softmax는 클래스 확률 계산(다중분류)에 주로 쓰임


#### 예측중에서 정확하게 예측한 것 추출
```python
images, labels, probs = get_predictions(model, test_iterator)
pred_labels = torch.argmax(probs, 1)
corrects = torch.eq(labels, pred_labels)
correct_examples = []

for image, label, prob, correct in zip(images, labels, probs, corrects):
    if correct:
        correct_examples.append((image, label, prob))

correct_examples.sort(reverse = True, key = lambda x: torch.max(x[2], dim = 0).values)
```


```
[.2 .8 .4 .4 .5]

np.max -> .8 (최대값)
np.argmax -> 1 (최대값의 인덱스)
```

#### 번외
```python
x = torch.randn([4,4])
print(x)

max_elements, max_idxs = torch.max(x, dim=0)
print(max_elements)
print(max_idxs)

'''
tensor([[-0.2984, 0.2491, -0.0098, -0.3784], 
		[-0.0150, 0.3422, 1.7460, -0.1150], 
		[-0.3944, -0.3097, 1.9204, 0.2835], 
		[ 1.7026, 1.3582, -0.2195, -0.3425]]) 
tensor([1.7026, 1.3582, 1.9204, 0.2835])
tensor([3, 3, 2, 2])
'''
```
- dim=0 행 기준으로 가장 큰 값

#### 이미지 출력을 위한 전처리
```python
def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image
```
- clamp는 주어진 min, max의 범주에 이미지가 위치하도록 함
- image.clamp_ 는 원본의 이미지를 수정하면서 반환하지 않는 조건
- image.add_ 는 이미지의 픽셀값을 정렬하기위한 연산
- 픽셀 값은 0이상의 값으로 세팅이됨
- (image_max - image_min + 1e-5)  최대에서 최소값을 빼고 엡실론을 더해서 ZeroDivision 방지
- Clamping이란 최댓값과 최솟값으로 범주화 시키는 것을 의미



```python
def plot_most_correct(correct, classes, n_images, normalize = True):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize = (25, 20))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)        
        image, true_label, probs = correct[i]
        image = image.permute(1, 2, 0)
        true_prob = probs[true_label]
        correct_prob, correct_label = torch.max(probs, dim = 0)
        true_class = classes[true_label]
        correct_class = classes[correct_label]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.cpu().numpy())
        ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n' \
                     f'pred label: {correct_class} ({correct_prob:.3f})')
        ax.axis('off')
        
    fig.subplots_adjust(hspace = 0.4)
```
- image.permute(1, 2, 0) 기존의 (채널, 행, 열)을 -> (행, 열, 채널)로 변경
- if normalize: 대체 행열


