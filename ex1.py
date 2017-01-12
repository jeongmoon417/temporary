#simple hello world using tensorflow

#create a constant op
#the op is added as a node to the default graph
#
#the valuse returned by the constructor represents the output of the constant op

import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")
sess = tf.Session()
print sess.run(hello)
