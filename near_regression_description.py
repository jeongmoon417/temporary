#r경사 하강법 알고리즘 : 매개변수(W, b)의 초기값에서 시작해서 아래 방법으로 W와 b를 수정해 가며 결국에는 오차함수를 최소화 하는변수 값을 찾아냄

#y=0.1*x + 0.3 의 그래프 생성
import numpy as np

num_points = 1000
vectors_set=[]

for i in xrange(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1*0.1+0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1,y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

#그림을 그리는 코드
import matplotlib.pyplot as plt

plt.plot(x_data, y_data, 'ro')
plt.legend()
plt.show()

#오차 함수 (= 비용함수) : W, b를 매개변수롤 받아 직선이 얼마나 데이터에 잘 맞는지를 기초로 하여 오차 값을 리턴
#평균제곱오차 : 실제 값과 알고리즘이 반복마다 추정한 값 사이의 거리를 오차로 하는 값의 평균
#Variable : 메서드 호출시 텐서플로 내부의 그래프 자료구조에 만들어질 하나의 변수를 정의
#loss(오차함수?) : 이미 알고 있는 값인 y_data 와 입력데이터 x_data에 의해 계산된 y값 사이의 거리를 제곱한 것의 평균을 계산
#오차 함수를 최소화하면 데이터에 가장 잘 들어맞는 모델을 얻을 수 있다!!
import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

#알고리즘을 텐서플로에서 사용하는 코드
#이 코드를 실행함므로써 텐서플로가 내부 자료구조(Variable인 W와 b) 안에 관련 데이터를 생성함
#이 옵티마이저는앞에서 정의한 비용함수에 경사 하강법 알고리즘을 적용
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#세션을 생성하고 run 메서드에 train 매개변수를 넣어 호출
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#훈련과정을 8번 반복
for step in xrange(8):
    sess.run(train)
    print step, sess.run(W), sess.run(b)
    print(step, sess.run(loss))

print sess.run(W), sess.run(b)
