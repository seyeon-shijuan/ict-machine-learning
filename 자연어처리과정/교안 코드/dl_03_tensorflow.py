import tensorflow as tf
import numpy as np


def make_nn():
    tf.random.set_seed(0)

    # 1. 뉴런 층 만들기
    input_layer = tf.keras.layers.InputLayer(input_shape=(3,))
    print(input_layer)

    hidden_layer = tf.keras.layers.Dense(units=4, activation='relu')
    print(hidden_layer)

    output_layer = tf.keras.layers.Dense(units=2, activation='softmax')

    # 2. 모델 만들기
    model = tf.keras.Sequential([
        input_layer,
        hidden_layer,
        output_layer
    ])

    print(model.summary())

    # 3. 컴파일
    model.compile(loss="mse", optimizer="Adam")
    print(input_layer.name, input_layer.dtype)
    print(model.layers[0].name)
    print(model.layers[1].name)

    # 4.
    print(f"{hidden_layer.activation.__name__ = }")
    print(f"{hidden_layer.weights = }")
    print(f"{output_layer.get_weights() = }")


def nn_training():
    # 1. 훈련 데이터
    x_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_train = np.array([[0], [1], [1]])

    # NN layer
    input_layer = tf.keras.layers.InputLayer(input_shape=(3,))
    hidden_layer = tf.keras.layers.Dense(units=4, activation="relu")
    output_layer = tf.keras.layers.Dense(units=2, activation="softmax")

    # 모델 생성
    model = tf.keras.Sequential([
        input_layer,
        hidden_layer,
        output_layer
    ])

    # compile
    model.compile(loss="mse", optimizer="Adam")

    # 은닉층의 정보
    inter_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)
    inter_layer_output = inter_layer_model(x_train)
    print(inter_layer_model)


if __name__ == '__main__':
    # make_nn()
    nn_training()

