import tensorflow as tf

# 데이터 준비
mnist = tf.keras.datasets.mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_train.ndim)

# 전처리: 0~255 사이의 픽셀의 값들을 0~1.0 사이의 값들로 변환
x_train, x_test = x_train/255.0, x_test/255.0

# 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# model.summary()

# 컴파일
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=5)

# 정확도 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"{test_loss = }, {test_acc = }")
