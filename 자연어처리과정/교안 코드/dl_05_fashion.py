import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 데이터 준비
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(f"{train_images.shape=}, {train_labels.shape=}")

# 시각화
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 'Sandal', "Shirt", "Sneaker", "Ankle boot"]

print(f"{train_labels[0]=}")
print(f"{train_images[0].shape=}")

plt.figure(figsize=(2, 2))
plt.imshow(train_images[0], cmap=plt.cm.binary)

# plt.show()

# 데이터 전처리: 정규화
train_images, test_images = train_images / 255.0, test_images / 255.0

# 모델 생성
model = tf.keras.models.Sequential([
    # 평탄화: 2차원 배열을 1차원으로 변환
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()

# 모델 컴파일
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 훈련
model.fit(train_images, train_labels, epochs=5)

# 정확도 평가
loss, acc = model.evaluate(test_images, test_labels)
print(f"{loss = }, {acc = }")

# 예측
y_pred = model.predict(test_images)
print(f"{y_pred=}")
print(np.argmax(y_pred[0]))

