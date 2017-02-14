#다중 계층 신경망 예제 : MNIST 이미지 셋을 사용하여 모델을 만드는 과정을 다중 계층 신경망을 이용하여 생성
#특징 : 거의 항상 입력 데이터로 이미지를 받음

#결과
#step 0, training accurancy 0.13
#step 100, training accurancy 0.91
#step 200, training accurancy 0.93
#step 300, training accurancy 0.95
#step 400, training accurancy 0.91
#step 500, training accurancy 0.97
#step 600, training accurancy 0.92
#step 700, training accurancy 0.95
#step 800, training accurancy 0.96
#step 900, training accurancy 0.97


from tensorflow.examples.tutorials.mnist import input_data

#MNIST 데이터를 로드
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf

#이미지 ?개와 ?개의 1~10사이의 수들
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#입력데이터를 원래 이미지의 구조로 재구성 : 2,3번째 차원은 입력의 너비와 높이, 4번째 차원은 컬러
x_image = tf.reshape(x, [-1,28,28,1])

#가중치
#truncated_normal :  Outputs random values from a truncated normal distribution
#truccated_normal : 잘린 정규분포로 부터 랜덤값을 출력
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#편향
#constant : Creates a constant tensor
#constant : 상수 텐서를 생성
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#합성곱 계층 : 시각적인 특징을 감지, 고유한 특징을 찾는다
#conv2d : computes a 2-D convolution given 4-D input and filter tensors
#conv2d : 주어진 4차원 입력과 필터 텐서(5X5의 필터)로 2차원 컨볼루션을 계산한다
#stride : 얼만클씩 옮겨갈 것인지 / padding : 이미지 외부를 채워서 끝까지 계산할 것인지 (SAME or VALID)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# 풀링계층 : 입력값을 단순하게 압축하고, 생산한 정보를 컴팩트한 버전으로 만든다
def max_pool_2X2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W_conv1=weight_variable([5,5,1,32])
#가중치 ----> 5X5: 윈도우 크기 / 1:컬러체널(흑백은 1) / 32: 얼마나 많은 특징을 사용할 것인지
#5X5의 W(필터)가 32개 임
b_conv1 = bias_variable([32])
#편향

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#relu 활성화 함수 " max(0, x)를 리턴한다
#shape= 1. VALID일 경우(?, 24, 24, 32) / 2. SAME일 경우(?, 28, 28, 32)
h_pool1 = max_pool_2X2(h_conv1)
#shpae=(?, 14, 14, 32)

#심층 신경망을 구성할 때는 여러 계층을 쌓아올릴 수 있다

W_conv2 = weight_variable([5,5,32,64])
#32: 이전 계층의 출력 값의 크기를 체널의 수로 넘긴다
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)    #shape=(?, 14, 14, 64)
h_pool2 = max_pool_2X2(h_conv2)     #shape=(?, 7, 7, 64) / 특징맵을 컴팩트 한것 64개

#소프트맥스 계층에 주입하기 위해 7X7 출력 값(h_pool2)을 완전 연결 계층에 연결
#전체 이미지를 처리하기 위해서는 1024개의 뉴런을 사용    ---1024는??????????????
w_fc1 = weight_variable([7*7*64, 1024])
#7X7크기의 결과값 64개(h_pool2)와, 우리가 임의로 선택한 뉴런의 수
b_fc1 = bias_variable([1024])

#이미지를 직렬화해서 벡터 형태로 입력
#가중치 행력 W_fc1과 일차원 벡터를 곱하고 편향 b_fc1을 더한 후 렐루 활성화 함수를 적용
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

#드롭아웃 : 신경망에서 필요한 매개변수 수를 줄이는 것 - 노드를 삭제하여 입력과 출력 사이의 연결을 제거하는 것
#어떤 뉴런을 제거하고 어떤것을 유지할 지는 무작위로 결정됨, 뉴런이 제거되거나, 그렇지 않을 확률은 코드처리 아니라 텐서플로에게 위임
#모델이 데이터에 오버피팅이 되는것을 막음 (오버피팅 : 뉴런의 수가 너무 많으면 오차가 너무 많아진다)
keep_prob = tf.placeholder("float")
#keep_prob : 뉴런이 드롭아웃 되지 않을 확률 저장
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#소프트맥스 계층
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#모델 훈련 및 평가 : ADAM 최적화 알고리즘
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accurancy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
        train_accuracy = sess.run(accurancy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print ("step %d, training accurancy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print ("test accuracy %g"% sess.run(accurancy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
