#!/usr/bin/python2.7
import tensorflow as tf
import numpy as np

A = np.zeros([1,3,3,5],dtype=np.float32)
B = np.zeros([2,2,5,1],dtype=np.float32)

for i in range(3):
  for j in range(3):
      A[:,i,j,:] = (i+j)/2.

tf_A = tf.Variable(A)

for i in range(5):
   B[:,:,i,:]=float(i)

tf_B = tf.Variable(B)

with tf.Session() as sess:
  init_op1,init_op2 = tf_A.initializer,tf_B.initializer
  sess.run(init_op1) ; sess.run(init_op2)
  print sess.run(tf_A) 
  print sess.run(tf_B) 
  op = tf.nn.conv2d(tf_A,tf_B,strides=[1,1,1,1],padding='SAME')
  print 'convolution operator start!'
  C = sess.run(op)
  print C
  print 'dim:',len(C),len(C[0]),len(C[0,0]),len(C[0,0,0])
  print '---------------------------'
