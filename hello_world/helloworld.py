#!/usr/bin/python2.7
import tensorflow as tf
import numpy as np

msg = 'hello~world'
sess = tf.Session()
var = tf.Variable(msg,str)
#constant(value, dtype=None, shape=None, name='Const')
PI = tf.constant(np.pi,dtype=np.float64,shape=[5],name='PI')
init=tf.initialize_all_variables()
sess.run(init)

print sess.run(var)
print 'PI : ',sess.run(PI)
