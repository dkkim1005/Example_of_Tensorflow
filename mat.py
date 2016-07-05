#!/usr/bin/python2.7
import numpy as np
import tensorflow as tf

A = np.array([[1,2],[2,3]],dtype=np.float64)
B = np.array([[1,2],[3,5]],dtype=np.float64)

tf_A = tf.Variable(A)
tf_B = tf.Variable(B)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

matmul_op = tf.matmul(A,B)

print 'A'
print sess.run(tf_A)
print 'B'
print sess.run(tf_B)

print 'tensorflow ver'
print 'A*B='
print sess.run(matmul_op)

print '-----------------'
print 'A*B='
print 'numpy ver'
print np.dot(A,B)

print '-----------------'
print 'copy tensor objects to numpy.'
C=tf_A.eval(sess)
print 'Type:',type(C)
print C

print '-----------------'
print 'summon initialization method after run tf.initialize_all_variables.'
tf_D = tf.Variable(10,dtype=np.float64)
#tf_D = tf_D.initialized_value()
init_op = tf_D.initializer
sess.run(init_op)
print sess.run(tf_D)

print '-----------------'
print 'assign'
tf_E = tf.Variable(1,dtype=np.float64)
tf_E.assign(tf_D)

#tf_E = tf_E.initialized_value()

init_op = tf_E.initializer
sess.run(init_op)
print sess.run(tf_E)

print tf_E
print tf_D

print '-----------------'
print 'use assign operation.'
tf_F = tf.Variable(30,dtype=np.float64)
init_op = tf_F.initializer
sess.run(init_op)
print 'init value:',sess.run(tf_F)
ass_op = tf.assign(tf_F,tf_E)
sess.run(ass_op)
print 'final value:',sess.run(tf_F)
