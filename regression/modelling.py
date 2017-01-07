#!/usr/bin/python2.7
import numpy as np
import tensorflow as tf
import sys

datSize = 1000
batchSize = 100
trainingSize = 900
testSize = datSize - trainingSize
learningRate = 1e-2
numEpoch = 10000
num_1_hidden_node = 100
num_2_hidden_node = 100
num_3_hidden_node = 100


x = 2.*np.pi*np.random.random([datSize,1]) - np.pi; x = x.astype('float32')
y = np.sin(x) + np.random.random()*1e-1 + np.exp(x)*np.cos(x) + x; y = y.astype('float32')

with open('sample.out','w') as f:
    for n in xrange(datSize):
        f.write('%f %f\n'%(x[n],y[n]))


tf_x = tf.placeholder('float32',[None,1])
tf_y = tf.placeholder('float32',[None,1])

# 1-hidden layer
m1 = tf.Variable(tf.random_uniform([1,num_1_hidden_node]),dtype='float32')
b1 = tf.Variable(tf.random_uniform([num_1_hidden_node]),dtype='float32')
h1 = tf.matmul(tf_x,m1) + b1
node1 = tf.nn.softmax(h1)

# 2-hidden layer
m2 = tf.Variable(tf.random_uniform([num_1_hidden_node,num_2_hidden_node]),dtype='float32')
b2 = tf.Variable(tf.random_uniform([num_2_hidden_node]),dtype='float32')
h2 = tf.matmul(node1,m2) + b2
node2 = tf.nn.softmax(h2)

# 3-hidden layer
m3 = tf.Variable(tf.random_uniform([num_2_hidden_node,num_3_hidden_node]),dtype='float32')
b3 = tf.Variable(tf.random_uniform([num_3_hidden_node]),dtype='float32')
h3 = tf.matmul(node2,m3) + b3
node3 = tf.nn.softmax(h3)

# model
mo = tf.Variable(tf.random_uniform([num_3_hidden_node,1]),dtype='float32')
bo = tf.Variable(tf.random_uniform([1]),dtype='float32')
model = tf.matmul(node3,mo) + bo

costFunction = tf.reduce_mean(tf.reduce_sum(tf.square(model - tf_y)/2.,reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(learningRate)

train = optimizer.minimize(costFunction)

with tf.Session() as sess:
    initVariable = tf.initialize_all_variables()
    sess.run(initVariable)
    numGroup = int(datSize/batchSize)
    numIter = int(numEpoch/datSize)
    sess.run(train,feed_dict={tf_x:x,tf_y:y})
    
    for niter in xrange(numIter):
        for n in xrange(numGroup):
            for i in xrange(batchSize):
                sess.run(train,feed_dict={\
                   tf_x:x[n*batchSize:(n+1)*batchSize,:],\
                   tf_y:y[n*batchSize:(n+1)*batchSize,:]})
            
            print "epoch:",(niter*datSize + (n+1)*batchSize),"cost:",sess.run(costFunction,feed_dict={\
                   tf_x:x[n*batchSize:(n+1)*batchSize,:],\
                   tf_y:y[n*batchSize:(n+1)*batchSize,:]})

    test_x = 2.*np.pi*np.random.random([datSize,1]) - np.pi; x = x.astype('float32')
    test_y = sess.run(model,feed_dict={tf_x:test_x})

    with open('test.out','w') as f:
        for n in xrange(datSize): 
            f.write('%f %f\n'%(test_x[n],test_y[n]))
