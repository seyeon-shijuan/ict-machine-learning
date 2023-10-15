import numpy as np
import matplotlib.pyplot as plt


def linear_regression():
    X = np.array([2, 4, 6, 8])
    y = np.array([81, 93, 91, 97])
    plt.figure(figsize=(4, 4))
    plt.scatter(X, y)

    # 기울기: (x-x평균)(y-y평균)의 합 / (x-x평균)**2
    frac1 = ((X - X.mean()) * (y - y.mean())).sum()
    frac2 = ((X - X.mean())**2).sum()
    a = frac1 / frac2
    print(f"기울기 a: {a}")

    # y의 절편b: y의 평균 - (x의 평균 * a)
    b = y.mean() - (X.mean() * a)
    print(f"y절편 b: {b}")

    y_pred = a * X + b
    print(f"예측값: {y_pred}")

    plt.plot(X, y_pred, 'r')

    print("="*30)
    a = 3
    b = 76
    y_pred = a * X + b
    print(f"기울기, 절편 수정 후 예측값: {y_pred}")

    # 에러 : 실제값 - 예측값
    error = y - y_pred
    print(error)

    # 평균제곱오차
    MSE = np.sum((error ** 2)) / len(X)
    print(MSE)
    # plt.show()


def sigmoid_test():
    # 8.0 까지 구하려고 8.1 작성(포함 아니니까)
    x = np.arange(-8, 8.1, 0.1)
    y = 1 / (1 + np.e ** (-x))

    plt.figure(figsize=(4, 4))
    plt.plot(x, y)

def sigmoid(x):
    return 1/(1 + np.e**(-x))


def logistic_regression():
    data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
    x_data = [i[0] for i in data]
    y_data = [i[1] for i in data]
    print("x:", x_data)
    print("x:", y_data)

    plt.scatter(x_data, y_data)

    a = b = 0
    lr = 0.05 # 학습률
    epochs = 2001 # 반복 횟수

    # 경사하강법
    for i in range(epochs):
        for x, y in data:
            # x * sigmoid(a * x + b) # x의 예측값
            a_tmp = x * (sigmoid(a * x + b) - y) # 오차
            b_tmp = sigmoid(a * x + b) - y # 마이너스 오차

            # 새로운 a, b의 값
            a = a - lr * a_tmp
            b = b - lr * b_tmp

            if i % 1000 == 0:
                print("epochs=%4d, a: %.2f, b: %.2f"%(i, a, b))

    print(f"a: {a}, b: {b}")
    print("실제값:", y_data)
    y_pred = [sigmoid(a * x + b) for x in x_data]
    print("예측값:", y_pred)

    x_range = np.arange(2, 14.1, 0.1)
    y_range = [sigmoid(a * x + b) for x in x_range]

    plt.plot(x_range, y_range, "r")

    plt.show()


def and_test(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    sigma = w1 * x1 + w2 * x2
    if sigma <= theta:
        return 0
    elif sigma > theta:
        return 1


def and_test2():
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]

    for x1, x2 in X:
        print(f"x1: {x1}, x2: {x2}")

    w1 = 0.5
    w2 = 0.5
    theta = 0.7 # 편향: 임계치

    print(w1 * 0 + w2 * 0 >= theta)
    print(w1 * 0 + w2 * 1 >= theta)
    print(w1 * 1 + w2 * 0 >= theta)
    print(w1 * 1 + w2 * 1 >= theta)

    for x1, x2 in X:
        ws = w1*x1 + w2*x2
        print("x1:", x1, "x2:", x2, "ws>=theta: ", ws >= theta)


def and_test3(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.7
    ws = w1 * x1 + w2 * x2 + b
    # print(sigma - theta >= 0) # -0.7
    if ws >= 0:
        return 1

    return 0


def and_gate(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.7
    ws = w1 * x1 + w2 * x2 + b
    if ws >= 0:
        return 1

    return 1


def or_gate(x1, x2):
    w1, w2, b = 0.5, 0.5, -0.2
    ws = w1 * x1 + w2 * x2 + b
    if ws >= 0:
        return 1

    return 0


def nand_gate(x1, x2):
    w1, w2, b = -0.5, -0.5, 0.7
    ws = w1 * x1 + w2 * x2 + b
    if ws >= 0:
        return 1

    return 0


def all_test():
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]

    print("x1  x2  AND  OR  NAND")
    print("=====================")
    print("0   0   {}   {}   {}".format(and_gate(0, 0), or_gate(0, 0), nand_gate(0, 0)))
    print("0   1   {}   {}   {}".format(and_gate(0, 1), or_gate(0, 1), nand_gate(0, 1)))
    print("1   0   {}   {}   {}".format(and_gate(1, 0), or_gate(1, 0), nand_gate(1, 0)))
    print("1   1   {}   {}   {}".format(and_gate(1, 1), or_gate(1, 1), nand_gate(1, 1)))
    print("=====================")

    # for x1, x2 in X:
    #     print("AND(%d,%d)=%d" % (x1, x2, and_gate(x1, x2)))
    #
    # for x1, x2 in X:
    #     print("OR(%d,%d)=%d" % (x1, x2, or_gate(x1, x2)))
    #
    # for x1, x2 in X:
    #     print("NAND(%d,%d)=%d" % (x1, x2, nand_gate(x1, x2)))



def AND(x):
    w = np.array([0.5, 0.5])
    b = -0.7

    ws = (x * w).sum() + b
    if ws >= 0:
        return 1

    return 0


def OR(x):
    w = np.array([0.5, 0.5])
    b = -0.2

    ws = (x * w).sum() + b
    if ws >= 0:
        return 1

    return 0


def NAND(x):
    w = np.array([-0.5, -0.5])
    b = 0.7

    ws = (x * w).sum() + b
    if ws >= 0:
        return 1

    return 0


def GATE(x, w, b):
    ws = (x * w).sum() + b
    if ws >= 0:
        return 1

    return 0

def xor_test():
    x_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    x_data = np.array(x_list)
    # print(x_data)

    for x in x_data:
        n1 = NAND(x)
        n2 = OR(x)
        s = np.array([n1, n2])
        y = AND(s)
        print("x1: %d, x2: %d, n1: %d, n2: %d, y: %d"%(x[0], x[1], n1, n2, y))


def xor_test2():
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # NAND 가중치    #OR 가중치
    w1 = np.array([[-0.5, -0.5], [0.5, 0.5]])
    b1 = np.array([0.7, -0.2])
    # AND 가중치
    w2 = np.array([[0.5, 0.5]])
    b2 = np.array([-0.7])

    for x in x_data:
        n1 = GATE(x, w1[0], b1[0])
        n2 = GATE(x, w1[1], b1[1])
        s = np.array([n1, n2])
        y = GATE(s, w2, b2)

        print("x1: %d, x2: %d, n1: %d, n2: %d, y: %d" % (x[0], x[1], n1, n2, y))




if __name__ == '__main__':
    # linear_regression()
    # sigmoid_test()
    # logistic_regression()
    # print(and_test(0, 0))
    # print(and_test(0, 1))
    # print(and_test(1, 0))
    # print(and_test(1, 1))
    # and_test2()
    # print(and_test3(0, 0))
    # print(and_test3(0, 1))
    # print(and_test3(1, 0))
    # print(and_test3(1, 1))
    # all_test()
    # xor_test()
    xor_test2()