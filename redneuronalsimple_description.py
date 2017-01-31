# ���ϰ��� �Ű�� ���� (MNIST �����ͼ�)
# ��� �ϰ���(Gradient Descent) �˰���� ������(backpropagation) �˰��� ���

from tensorflow.examples.tutorials.mnist import input_data

#�Ʒ� �����Ͱ� ����ִ� mnist.train�� / �׽�Ʈ �����Ͱ� ����ִ� mnist.test�� ����
#�Ʒ� �̹����� ���� : mnist.train.image / ���̺��� ���� : mnist.train.labels
#�̹����� 28*28(=784)�� �����Ǿ� ����, 0~1������ ���ڷ� ���� ������ ǥ��
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# �� ���� : x=�̹���(x���� ���� ������ ����), W=����ġ, b=����, y=Ȯ����
x = tf.placeholder("float", [None, 784])
#None: � ũ�⳪ �����ϴ� -- �̹����� ����
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#placeholer : �� �Ʒ�(train)�� �ϸ鼭 �ٲ��� �ʴ� �Է� ������ ����, �ʱⰪ ������ �ʿ� ����
#variable : �� �Ʒ�(train)�� �ϸ鼭 �Ʒõ� �����͵��� ����(�ٲ��), �ʱⰪ ������ �ʿ� ��


matm = tf.matmul(x, W)
y = tf.nn.sotfmax(tf.matmul(x, W) + b)
#matmul(x, W) : ����� ����
#softmax(mat) : �� ī�װ�(0~9)�� ���� Ȯ�� ������ ���� 

# ���� ��Ʈ����(cross entropy)�Լ��� �����ϱ� ���� �÷��̽� Ȧ��
# ���� ��Ʈ����(y' = ���� ���� / y = ������ Ȯ������) = -sum(y'*log(y))
y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(_y*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#�н� �ӵ� 0.01�� ����ϰ��� �˰����� ����Ͽ� ũ�ν� ��Ʈ���Ǹ� �ּ�ȭ�ϴ� ������ �˰����� ���
#������ �˰��� : �ټ��÷ΰ� ���� �Ʒý�Ű�� ���� ������ ����Լ��� ���⸦ ã�� ����ȭ �˰���

sess = tf.Session()
sess.run(tf.initialize_all_variables())
#init_op= tf.global_variables_initializer()

#�� �Ʒ�
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #�Ʒ� �����ͼ����κ��� �������� 100���� ����
    sess.run(train_step, feed_dict={x: vatch_xs, y_:batch_ys})
    #�÷��̽�Ȧ���� ����Ͽ� �� 100���� ���� �����͸� ����

#�� ��
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#argmax(input, dimension, name=None) : input�� �ټ����� dimention���������� �ִ밪�� ���� ���̺��� ����
#correct_prediction �� �Ҹ���(boolean)�� ����
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#cast (c, "float") : �Ǽ������� c�� ����ȯ
#reduce_mean() : ��հ� ���
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.lables})
