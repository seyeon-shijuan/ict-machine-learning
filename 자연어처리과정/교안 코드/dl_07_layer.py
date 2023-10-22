import tensorflow as tf
import numpy as np

# https://colab.research.google.com/drive/1jaIBNQsA7p6qTCNcmyPr_UhJNwVOcCwa?usp=sharing#scrollTo=q3l_RMlE0yZE

#train data : XOR
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

x_train = np.array(x_data, dtype=np.float32)
y_train = np.array(y_data, dtype=np.float32)
print(x_train, y_train)

#Dense layer 구현: 2 Layer
model = tf.keras.Sequential([
    #(None, 2)
    #(m,n) * (n*l) = (m,l) #내적의 곱
    #(4,2) * (2 , 2) = (4, 2)
    #Param : 6 (w : 4, b: 2) :
    # w : w의 shape값의 곱, b = w의 컬럼의 개수(2차원 경우)
    #파라미터 수: (입력층의 뉴런수 + 1) * (출력층의 뉴런수 ) = (2+1)*2 = 6

    tf.keras.layers.Dense(units= 2, input_shape=(2,), activation="sigmoid"),

    #(m,n) * (n*l) = (m,l) #내적의 곱
    #(4,2) * (2 , 1) = (4, 1)

    tf.keras.layers.Dense(units=1, activation="sigmoid")

])

model.summary()

#컴파일
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              metrics=["accuracy"]) #학습에 사용하는 파라미터 설정

#기계학습
model.fit(x_train, y_train, epochs=1000, verbose=1)

x_train = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0],
                   [11.0,12.0,13.0,14.0,15.0]]) #(3,5)

y_train = np.array([[1.1, 2.2, 3.3, 4.4, 5.5],
                   [6.1, 7.2, 8.3, 9.4, 10.5],
                   [11.1, 12.2, 13.3, 14.4, 15.5]]) #(3,5)

print(x_train)
print(y_train)

tf.random.set_seed(5) #변화의 폭을 일정하게 하기위해서 랜덤시드를 설정


#모델 생성: 4 layer

model = tf.keras.Sequential([
    #(3,5) * (5,100 ) = (3,100)
    tf.keras.layers.Dense(units=100, input_shape=(5,), activation="relu"),

    #(3,100) * (100,50) = (3, 50)
    tf.keras.layers.Dense(units=50, activation="relu"),

    #(3,50) * (50,25) = (3,25)
    tf.keras.layers.Dense(units=25, activation="relu"),

    #(3,25) * ( 25,5) = (3,5)
    tf.keras.layers.Dense(units=5)

])
