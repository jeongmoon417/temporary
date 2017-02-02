from tensorflow.examples.tutorials.mnist import input_data

#MNIST 데이터를 로드
mnist = input_data.read_Data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#입력데이터를 원래 이미지의 구조로 재구성 : 2,3번째 차원은 입력의 너비와 높이, 4번째 차원은 컬러
x_image = tf.reshape(x, [-1,28,28,1]
