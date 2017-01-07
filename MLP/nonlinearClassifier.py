#!/usr/bin/python2.7
import numpy as np
import tensorflow as tf
import sys 
from file_reader import convertData

try:
    fileNameRed = sys.argv[1]
    fileNameBlue = sys.argv[2]

except:
    print "argv[1]: file name to read data(Red)."
    print "argv[2]: file name to read data(Blue)."
    exit()


dataTot, dataTag = convertData(fileNameRed,fileNameBlue)


# MLP construction
n1_node = 10; n2_node = 10; n3_node = 10
learningRate = 0.01
numEpoch = 1000

input_vector = tf.placeholder('float32',[None,2])
target_vector = tf.placeholder('float32',[None,2])

w1 = tf.Variable(tf.random_normal([2,n1_node]),dtype='float32')
b1 = tf.Variable(tf.random_normal([n1_node]),dtype='float32')
layer_1 = tf.nn.softmax(tf.matmul(input_vector,w1) + b1) 

w2 = tf.Variable(tf.random_normal([n1_node,n2_node]),dtype='float32')
b2 = tf.Variable(tf.random_normal([n2_node]),dtype='float32')
layer_2 = tf.nn.softmax(tf.matmul(layer_1,w2) + b2) 

w3 = tf.Variable(tf.random_normal([n2_node,n3_node]),dtype='float32')
b3 = tf.Variable(tf.random_normal([n3_node]),dtype='float32')
layer_3 = tf.nn.softmax(tf.matmul(layer_2,w3) + b3) 

w4 = tf.Variable(tf.random_normal([n3_node,2]),dtype='float32')
b4 = tf.Variable(tf.random_normal([2]),dtype='float32')
layer_4 = tf.nn.softmax(tf.matmul(layer_3,w4) + b4) 

cross_entropy = target_vector*tf.log(layer_4)

cost_function = tf.reduce_mean(-tf.reduce_sum(cross_entropy,reduction_indices=1))

optimizer = tf.train.AdamOptimizer(learningRate)

train = optimizer.minimize(cost_function)


with tf.Session() as sess:
    model = tf.initialize_all_variables()
    sess.run(model)
    for n in xrange(numEpoch):
        sess.run(train,feed_dict={input_vector:dataTot,target_vector:dataTag})
        print "# of epoch:",n,"cost:",sess.run(cost_function,feed_dict={input_vector:dataTot,target_vector:dataTag})
    result = sess.run(layer_4,feed_dict={input_vector:dataTot})

print result
                                                                          61,1          Bot
