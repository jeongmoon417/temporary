# 단일계층 신경망 예제 (MNIST 데이터셋)
# 경사 하강법(Gradient Descent) 알고리즘과 역전파(backpropagation) 알고리즘 사용

from tensorflow.examples.tutorials.mnist import input_data

#훈련 데이터가 들어있는 mnist.train과 / 테스트 데이터가 들어있는 mnist.test를 얻음
#훈련 이미지의 참조 : mnist.train.image / 레이블의 참조 : mnist.train.labels
#이미지는 28*28(=784)로 구성되어 있음, 0~1사이의 숫자로 검은 정도를 표시
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# 모델 생성 : x=이미지(x점에 대한 정보를 저장), W=가중치, b=편향, y=확률값
x = tf.placeholder("float", [None, 784])
#None: 어떤 크기나 가능하다 -- 이미지의 갯수
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#placeholer : 모델 훈련(train)을 하면서 바뀌지 않는 입력 데이터 저장, 초기값 설정이 필요 없음
#variable : 모델 훈련(train)을 하면서 훈련될 데이터들을 저장(바뀐다), 초기값 설정이 필요 있


matm = tf.matmul(x, W)
y = tf.nn.sotfmax(tf.matmul(x, W) + b)
#matmul(x, W) : 행렬의 곱셈
#softmax(mat) : 각 카테고리(0~9)에 속할 확률 분포를 리턴 

# 교차 엔트로피(cross entropy)함수를 구현하기 위한 플레이스 홀더
# 교차 엔트로피(y' = 실제 분포 / y = 에측된 확률분포) = -sum(y'*log(y))
y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(_y*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#학습 속도 0.01과 경사하강법 알고리즘을 사용하여 크로스 엔트로피를 최소화하는 역전파 알고리즘을 사용
#역전파 알고리즘 : 텐서플로가 모델을 훈련시키기 위해 적절한 비용함수의 기울기를 찾는 최적화 알고리즘

sess = tf.Session()
sess.run(tf.initialize_all_variables())
#init_op= tf.global_variables_initializer()

#모델 훈련
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #훈련 데이터셋으로부터 무작위로 100개를 추출
    sess.run(train_step, feed_dict={x: vatch_xs, y_:batch_ys})
    #플레이스홀더를 사용하여 이 100개의 샘플 데이터를 주입

#모델 평가
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#argmax(input, dimension, name=None) : input의 텐서에서 dimention차원에서의 최대값을 가진 레이블을 리턴
#correct_prediction 은 불리언(boolean)을 리턴
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#cast (c, "float") : 실수형으로 c를 형변환
#reduce_mean() : 평균값 계산
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.lables})
