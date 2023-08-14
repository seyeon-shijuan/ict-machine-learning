import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


print("Num GPUs available: ", len(tf.config.experimental.list_physical_devices()))


x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
print(x)

v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
print(v, end="\n ==================")

v.assign(tf.ones((3, 1)))
print(v)

v[0, 0].assign(3.)

# 텐서 연산
a = tf.ones((2, 2))
b = tf.square(a)

# gradient api
# time = tf.Variable(0.)

# with tf.GradientTape() as outer_tape:
#     with tf.GradientTape() as inner_tape:
#         position = 4.9 * time ** 2
    
#     speed = inner_tape.gradient(position, time)

# acceleration = outer_tape.gradient(speed, time)

# tmp = 'here'

# 텐서플로 선형 분류기

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)


inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                    np.ones((num_samples_per_class, 1), dtype="float32")))

# plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[: ,0])
# plt.show()

# 선형 분류기의 변수 만들기
input_dim = 2
output_dim = 1

W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

# 정방향 패스 함수
def model(inputs):
    return tf.matmul(inputs, W) + b

# 평균 제곱 오차 손실 함수
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)

# 훈련 스텝 함수
learning_rate = 0.1

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)

    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)

    return loss

# 배치 훈련 룹
for step in range(40):
    loss = training_step(inputs, targets)
    print(f"{step} 번째 스텝의 손실: {loss:.4f}")


predictions = model(inputs)
# plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
# plt.show()

x = np.linspace(-1, 4, 100)
y = - W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()

