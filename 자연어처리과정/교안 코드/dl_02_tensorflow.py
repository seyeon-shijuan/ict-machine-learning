import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def scalar_test():
    # tensorflow에서 scalar 만들기
    scalar = tf.constant(1)
    print(scalar, ":", tf.rank(scalar))

    # tensorflow에서 vector 만들기
    vector = tf.constant([1, 2, 3])
    print(vector, ":", tf.rank(vector))

    # tensorflow에서 matrix 만들기
    matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(matrix, ":", tf.rank(matrix))

    tensor = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                          [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    print(tensor, ":", tf.rank(tensor))


def scalar_test2():
    a = tf.constant(1)
    b = tf.constant([2])
    c = tf.constant([[1, 2], [3, 4]])
    print(a)
    print(b)
    print(c)
    print("="*30)

    a = tf.zeros(1)
    b = tf.zeros([2])
    c = tf.zeros([2, 3])
    print(a)
    print(b)
    print(c)
    print("=" * 30)

    a = tf.ones(3)
    b = tf.ones([4])
    c = tf.ones([2, 2, 2])
    print(a)
    print(b)
    print(c)
    print("=" * 30)


def scalar_test3():
    a = tf.range(0, 3)
    b = tf.range(1, 5, 2)
    print(a)
    print(b)
    print("=" * 30)

    a = tf.linspace(0, 1, 3)
    b = tf.linspace(0, 3, 10)
    print(a)
    print(b)

    print(a.dtype, a.shape)
    print(b.dtype, b.shape)


def scalar_test4():
    a = tf.constant([1, 2, 3])
    print(a, type(a))
    print(a.numpy()) # 형변환 : Tensor -> ndarray


def model_test():
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(loss="mean_squared_error", optimizer="sgd")

    x = np.array([-1.0, 0, 1.0, 2.0, 3.0, 4.0]) # 기본이 float임
    print(x.dtype) # float64
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    print(y)

    model.fit(x, y, epochs=50)

    y_pred = model.predict([5.0])
    print(y_pred)


def model_test2():
    x = np.array([-1.0, 0, 1.0, 2.0, 3.0, 4.0])  # 기본이 float임
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # 입력 1개, 출력 3개인 NN
    model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])
    model.compile(loss="mse", optimizer="sgd")
    model.fit(x, y, epochs=500)
    y_pred = model.predict([0])
    print(y_pred)

    model.evaluate([0], [[0, 1, 0]])


def model_test3():
    # 입력 1개, 출력 3개인 NN
    model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])
    model.compile(loss="mse", optimizer="SGD")
    hist = model.fit([1], [[0, 1, 0]], epochs=100)
    print(hist)
    print("history:", hist.history)
    loss = hist.history['loss']
    print("loss:", loss)

    plt.figure(figsize=(4, 3))
    plt.plot(loss)
    plt.show()

    print('='*30)
    y_pred = model.predict([0])
    print(y_pred)

    model.evaluate([0], [[0, 1, 0]])


def optimizer_test():
    tf.random.set_seed(0)
    model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])
    print(model)

    tf.random.set_seed(0)
    model2 = tf.keras.models.clone_model(model)
    print(model2)

    tf.random.set_seed(0)
    model3 = tf.keras.models.clone_model(model)
    print(model3)

    model.compile(loss="mse", optimizer="SGD")
    model2.compile(loss="mse", optimizer="Adam")
    model3.compile(loss="mse", optimizer="RMSprop")

    history1 = model.fit([1], [[0, 1, 0]], epochs=100, verbose=0)
    history2 = model.fit([1], [[0, 1, 0]], epochs=100, verbose=0)
    history3 = model.fit([1], [[0, 1, 0]], epochs=100, verbose=0)

    loss1 = history1.history["loss"]
    loss2 = history2.history["loss"]
    loss3 = history3.history["loss"]

    plt.figure(figsize=(4, 3))
    plt.plot(loss1, label="SGD")
    plt.plot(loss2, label="Adam")
    plt.plot(loss3, label="RMSprop")
    plt.legend()

    plt.show()


def and_with_tf():
    # 1. 훈련 데이터 준비
    x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [[0], [0], [0], [1]]

    # 2. model
    model = keras.Sequential([
        keras.layers.Dense(units=3, input_shape=[2], activation="relu"),
        keras.layers.Dense(units=1)
    ])

    model.summary()

    # 3. compile 딥러닝 환경 설정
    model.compile(loss="mse", optimizer="adam")

    # 4. train 학습
    y_pred1 = model.predict(x_train)
    print("학습 전: ", y_pred1)

    history = model.fit(x_train, y_train, epochs=1000, verbose=0)

    # 5. predict 예측
    y_pred2 = model.predict(x_train)
    print("학습 후: ", y_pred2)

    # 6. loss 확인
    loss = history.history["loss"]

    plt.figure(figsize=(4, 3))

    plt.plot(loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # scalar_test()
    # scalar_test2()
    # scalar_test3()
    # scalar_test4()
    # model_test()
    # model_test2()
    # model_test3()
    # optimizer_test()
    and_with_tf()