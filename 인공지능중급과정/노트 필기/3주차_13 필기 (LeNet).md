
#### LeNet-5

![](https://blog.kakaocdn.net/dn/bvndfd/btr92v4mDpH/AUWzSw5MSO2TLm1eSURGQ1/img.png)

```python
class ImageTransform():    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img)
```

ⓐ transforms.Compose: 이미지를 변형할 수 있는 방식들의 묶음
ⓑ transforms.RandomResizedCrop: 입력 이미지를 주어진 크기(resize: 224×224)로 조정
또한, scale은 원래 이미지를 임의의 크기(0.5~1.0(50~100%))만큼 면적을 무작위로 자르겠다는 의미
ⓒ transforms.RandomHorizontalFlip: 주어진 확률로 이미지를 수평 반전시킴
- 이때 확률 값을 지정하지 않았으므로 기본값인 0.5의 확률로 이미지들이 수평 반전
- 즉, 훈련 이미지 중 반은 위아래 뒤집힌 상태로 두고, 반은 그대로 사용
ⓓ transforms.ToTensor: ImageFolder 메서드를 비롯해서 torchvision 메서드는 이미지를 읽을 때 파이썬 이미지 라이브러리인 PIL을 사용
- PIL을 사용해서 이미지를 읽으면 생성되는 이미지는 범위가 [0, 255]이며, 배열의 차원이 (높이 H×너비 W×채널 수 C)로 표현
- 이후 효율적인 연산을 위해 torch.FloatTensor 배열로 바꾸어야 하는데, 이때 픽셀 값의
- 범위는 [0.0, 1.0] 사이가 되고 차원의 순서도 (채널 수 C×높이 H×너비 W)로 바뀜
- 이러한 작업을 수행해 주는 메서드가 ToTensor()
ⓔ transforms.Normalize: 전이 학습에서 사용하는 사전 훈련된 모델들은 대개 ImageNet 데이터셋에서 훈련
- 사전 훈련된 모델을 사용하기 위해서는 ImageNet 데이터의 각 채널별 평균과 표준편차에 맞는 정규화(normalize)를 해 주어야 함
- 즉, Normalize 메서드 안에 사용된 (mean: 0.485, 0.456, 0.406), (std: 0.229, 0.224, 0.225)는 ImageNet에서 이미지들의 RGB 채널마다 평균과 표준편차를 의미
- 참고로 OpenCV를 사용해서 이미지를 읽어 온다면 RGB 이미지가 아닌 BGR 이미지이므로 채널 순서에 주의해야 함



```python
cat_directory = 'catanddog/train/Cat/'
dog_directory = 'catanddog/train/Dog/'

cat_images_filepaths = sorted([os.path.join(cat_directory, f) for f in os.listdir(cat_directory)])   
dog_images_filepaths = sorted([os.path.join(dog_directory, f) for f in os.listdir(dog_directory)])
images_filepaths = [*cat_images_filepaths, *dog_images_filepaths]    
correct_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None]    

random.seed(42)    
random.shuffle(correct_images_filepaths)
#train_images_filepaths = correct_images_filepaths[:20000] #성능을 향상시키고 싶다면 훈련 데이터셋을 늘려서 테스트해보세요   
#val_images_filepaths = correct_images_filepaths[20000:-10] #훈련과 함께 검증도 늘려줘야 합니다
train_images_filepaths = correct_images_filepaths[:300]    
val_images_filepaths = correct_images_filepaths[300:-10]  
test_images_filepaths = correct_images_filepaths[-10:]    
print(len(train_images_filepaths), len(val_images_filepaths), len(test_images_filepaths))
```

- cat_images_filepaths: 불러와서 정렬
- images_filepaths = [*cat_images_filepaths, *dog_images_filepaths] : asterisk는 전체인자값가져오기
- correct_images_filepaths: cv에서 image를 read해서 올바른 것들만 list로 추림
- train_images_filepaths 섞은 것에서 300개 추출하여 10개분리
- 300 75 10


```python
def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
        predicted_label = predicted_labels[i] if predicted_labels else true_label
        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
```

```python
display_image_grid(test_images_filepaths)
```

- rows = len(images_filepaths) // cols 행 개수 계산
- figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
- image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) : convert color BGR에서 RGB로 바꿈
- true_label = os.path.normpath(image_filepath).split(os.sep) -2 : 
- catanddog/train/Dog/999.jpg 에서 -1은 999.jpg, -2는 Dog <- 라벨값을 붙이려고 가져옴
- predicted_label = predicted_labels[i] if predicted_labels else true_label
- ax.ravel()[i].imshow(image) ax 기준으로 idx번째에 그림 채우기
- plt.tight_layout() 이미지 여백 조정

김광석 - 혼자 남은 밤
https://www.youtube.com/watch?v=S5dcSZXdc7c
을 갑자기 추천해주심.. 🎶


```python
class DogvsCatDataset(Dataset):    
    def __init__(self, file_list, transform=None, phase='train'):    
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):       
        img_path = self.file_list[idx]
        img = Image.open(img_path)        
        img_transformed = self.transform(img, self.phase)
        
        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0
        return img_transformed, label
    
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 32

train_dataset = DogvsCatDataset(train_images_filepaths, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = DogvsCatDataset(val_images_filepaths, transform=ImageTransform(size, mean, std), phase='val')

index = 0
print(train_dataset.__getitem__(index)[0].size())
print(train_dataset.__getitem__(index)[1])
```


```python
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

batch_iterator = iter(train_dataloader)
inputs, label = next(batch_iterator)
print(inputs.size())
print(label)
```


### LeNet

```python
class LeNet(nn.Module):
```


