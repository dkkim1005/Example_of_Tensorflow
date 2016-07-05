#!/usr/bin/python2.7
import numpy as np
import tensorflow as tf

'''
**Important**: This tensor will produce an error if evaluated. Its value must
be fed using the `feed_dict` optional argument to `Session.run()`,
`Tensor.eval()`, or `Operation.run()`.
'''

x = tf.placeholder(tf.float32,shape=(10,10))
sess = tf.Session()
y = tf.matmul(x,x)

rand_mat = np.random.rand(10,10)
print sess.run(y,feed_dict={x: rand_mat})

print '---------------'

z = tf.placeholder(tf.float64,shape=(None,None))

rand_mat2 = np.random.random([4,5])
y = tf.matmul(z,z,transpose_a=True)

print sess.run(y,feed_dict={z: rand_mat2})

print '---------------'
print np.dot(rand_mat2.T,rand_mat2)


A = sess.run(y,feed_dict={z: rand_mat2})
print '---------------'
print A
