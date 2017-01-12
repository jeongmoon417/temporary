#basic operation example using tensorflow library

#1. basic constant operations
import tensorflow as tf
a=tf.constant(2)
b=tf.constant(3)

with tf.Session() as sess:
  print "a=2, b=3"        
  print "Addition with constants %i" % sess.run(a+b)        
  print "Multiplication with constants: %i" % sess.run(a*b)
  

#2. basic operations with variable as graph input

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a,b)
mul = tf.mul(a,b)with tf.Session() as sess:        
  print "Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3})        
  print "Multiblication with variables: %i" % sess.run(mul, feed_dict={a: 2,b: 3})
