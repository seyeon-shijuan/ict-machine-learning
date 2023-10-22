import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# https://colab.research.google.com/drive/1KU3DZIDbAuYLkrEVsjgTgqzXYs2UMe3A?usp=sharing#scrollTo=asOgOgOJ4kZK

def number_cnn_test():
    # 입력 이미지
    image = np.array([[[[1], [2], [3]],
                       [[4], [5], [6]],
                       [[7], [8], [9]]]], dtype =np.float32)

    print(image)
    print(image.shape) # (1(사진 개수), 3, 3, 1(컬러: 1(흑백), 3(컬러:)))

    print(image.reshape(3, 3)) # 2차원으로 변경
    # plt.imshow(image.reshape(3, 3), cmap="Greys")
    # plt.show()

    # conv2d layer, filter : (2, 2, 1, 1)      , strides: 1
    # 가로 세로 인풋 채널 아웃풋 채널
    # filter: (2, 2, 1, 1) -> 가로, 세로, 컬러, 필터의 개수(출력 개수)
    weight = tf.constant([[[[1.]], [[1.]]],
                          [[[1.]], [[1.]]]])
    print(weight)

    # 출력 이미지
    # (N - F) / strides + 1
    # (3 - 2) / (1+1)
    # (1, 3, 3, 1) -> (1, 2, 2, 1)

    conv2d = tf.nn.conv2d(input=image, filters=weight, strides=[1, 1, 1, 1], padding='VALID')
    conv2d_image = conv2d.numpy()

    print(conv2d_image)

    plt.imshow(conv2d_image.reshape(2, 2), cmap="Greys")
    plt.show()

    ######################
    #filter : ( 2, 2, 1, 1)
    weight = tf.constant([[[[1.]], [[1.]]],
                          [[[1.]], [[1.]]]])

    print(weight)

    conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="SAME")
    conv2d_image = conv2d.numpy()
    print(conv2d_image.shape) #(1,2,2,1)

    plt.imshow(conv2d_image.reshape(3, 3), cmap="Greys")
    plt.show()

'''
mnist의 경우
첫 번째 conv2 layer
원본 이미지: (None, 28, 28, 1) = 채널, 가로, 세로, 필터
필터: (3, 3, 1, 32), strides : (1, 1, 1, 1)
(28+2 -3(filter)) / 1+1 = 28
'''

image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], dtype=np.float32)

# conv2d layer: 3 filter padding = "SAME"
# image: (1, 3, 3, 1(입력)), filter: (2, 2, 1, 3(출력)), strides: (1, 1, 1, 1)
# 입력: 1 ---> 출력: 3

weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1]]],
                      [[[1.0,10.,-1.]],[[1.,10.,-1]]]])
print(weight)
# print(weight.numpy())
# print(weight.shape)

conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="SAME")
print(conv2d)

conv2d_image = conv2d.numpy()
print(conv2d_image.shape)

