#!/usr/bin/python2.7
import numpy as np
import tensorflow as tf

nSize = 1000
learningRate = 0.01
nEpoch = 1000
red_x = np.random.random([nSize]).astype('float32')
red_y = np.random.random([nSize]).astype('float32')

blue_x = np.random.random([nSize]).astype('float32') + 5.
blue_y = np.random.random([nSize]).astype('float32') + 5.

data_set = np.zeros([nSize*2,2],dtype='float32')
data_set[:nSize,0] = red_x[:]
data_set[:nSize,1] = red_y[:]
data_set[nSize:,0] = blue_x[:]
data_set[nSize:,1] = blue_y[:]

tag = np.zeros([nSize*2,2])
tag[:nSize,0] = 0.
tag[:nSize,1] = 1.
tag[nSize:,0] = 1. 
tag[nSize:,1] = 0. 

# perceptron model construnction.

input_vector = tf.placeholder('float32',[None,2])
target_vector = tf.placeholder('float32',[None,2])

w = tf.Variable(tf.random_uniform([2,2]),dtype='float32')
b = tf.Variable(tf.random_uniform([2]),dtype='float32')

output_vector = tf.nn.softmax(tf.matmul(input_vector,w) + b)

cross_entropy = target_vector*tf.log(output_vector)

cost_function = tf.reduce_mean(-tf.reduce_sum(cross_entropy,reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learningRate)

train = optimizer.minimize(cost_function)

# perform learning

sess = tf.Session()
model = tf.initialize_all_variables()
sess.run(model)
for n in xrange(1,nEpoch+1):
    sess.run(train,feed_dict={input_vector:data_set,target_vector:tag})
    if(n%100 == 0 and n != 0):
        print "# of epoch",n,"cost:",sess.run(cost_function,feed_dict={input_vector:data_set,target_vector:tag})


# test model

test_red_x = np.random.random([nSize]).astype('float32')
test_red_y = np.random.random([nSize]).astype('float32')

test_blue_x = np.random.random([nSize]).astype('float32') + 5.
test_blue_y = np.random.random([nSize]).astype('float32') + 5.

test_data_set = np.zeros([nSize*2,2],dtype='float32')
test_data_set[:nSize,0] = test_red_x[:]
test_data_set[:nSize,1] = test_red_y[:]
test_data_set[nSize:,0] = test_blue_x[:]
test_data_set[nSize:,1] = test_blue_y[:]

test_tag = np.zeros([nSize*2,2])
test_tag[:nSize,0] = 0.
test_tag[:nSize,1] = 1.
test_tag[nSize:,0] = 1. 
test_tag[nSize:,1] = 0. 


result = sess.run(output_vector,feed_dict={input_vector:test_data_set})

accuracy = 0.
for n in xrange(nSize):
    if(result[n,1] > 0.5):
        accuracy += 1./(2*nSize)
    if(result[n+nSize,0] > 0.5):
        accuracy += 1./(2*nSize)

print "accuracy:",accuracy
