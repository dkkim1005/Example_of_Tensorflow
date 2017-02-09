#!/usr/bin/python2.7
import numpy as np
import tensorflow as tf


def inLayerGenerator(outLayer, outNumDim, inNumDim, dtype='float32'):
    # Generates next layer with normalized signal.

    """
    neuron  out   xW     in    neuron
       0    ---    X    ---     0
       0    ---    X    ---     0
       ..          X    ---     ..
       0    ---    X    ---     0
       0    ---    X    ---     0
                  +b
    """
    W = tf.Variable(tf.random_uniform([outNumDim,inNumDim]),dtype=dtype)
    b = tf.Variable(tf.random_uniform([inNumDim]),dtype=dtype)

    inLayer = tf.matmul(tf.cast(outLayer,dtype),W) + b

    return inLayer


def multiLayerGenerator(inputLayer, numNodeList, active = tf.sigmoid, prob_dropout = 1):
    # The # of hidden layer should be larger than 1.
    assert(len(numNodeList) >= 3)    

    _outLayer = inputLayer
    size = len(numNodeList)

    # Connect hidden layer
    for i in range(size-2):
        outNumDim = numNodeList[i]
        inNumDim = numNodeList[i+1]
        _inLayer = inLayerGenerator(_outLayer,outNumDim,inNumDim)
        _outLayer = active(_inLayer)
        _outLayer = tf.nn.dropout(_outLayer,prob_dropout)

    finOutNumDim = numNodeList[-2];
    finInNumDim = numNodeList[-1];

    totLayer = inLayerGenerator(_outLayer, finOutNumDim, finInNumDim)

    return totLayer


class dataInsertion:
    def __init__(self):
        self.inData = []
        self.outData = []
        self.numTail = -1


    def insert_data(self, inData, outData):
        size_col = len(inData)

        assert(size_col == len(outData))

        for i in xrange(size_col):
            self.inData.append([])
            self.outData.append([])
            self.numTail += 1

            for j,datum in enumerate(inData[i]):
                self.inData[self.numTail].append(datum)

            for j,datum in enumerate(outData[i]):
                self.outData[self.numTail].append(datum)

        self.inData = np.array(self.inData)
        self.outData = np.array(self.outData)


class ANN_Regression(dataInsertion):
    def __init__(self, learning_rate, numNodeList, active = tf.nn.softmax, prob_dropout = 1.0, dtype='float32'):
        dataInsertion.__init__(self)
        inNumDim = numNodeList[0]
        outNumDim = numNodeList[-1]

        self.inLayer = tf.placeholder(dtype,[None,inNumDim])
        self.outLayer = tf.placeholder(dtype,[None,outNumDim])
        self.totLayer = multiLayerGenerator(self.inLayer, numNodeList, active, prob_dropout)

        self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.outLayer - self.totLayer),reduction_indices=[1]))

        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


    def run(self, sess, numIter, batchSize=1.):
        assert(isinstance(sess,tf.Session))

        totSize = self.numTail + 1

        cutSize = int(totSize * batchSize)

        for niter in xrange(numIter):
            arr = range(totSize-1)
            arr = np.random.permutation(arr)
            arr = arr[:cutSize]
            sess.run(self.train,feed_dict={self.inLayer : self.inData[arr], self.outLayer : self.outData[arr]})
            if(niter % 10 == 0):
                print "iter:",niter,"cost( |y - y'|^2 ):",\
                      sess.run(self.cost,feed_dict={self.inLayer : self.inData, self.outLayer : self.outData})

        print "cost( |y - y'|^2 ):",sess.run(self.cost,feed_dict={self.inLayer : self.inData, self.outLayer : self.outData})


    def predict(self, sess, x):
        return sess.run(self.totLayer,feed_dict={self.inLayer : x})


if __name__ == "__main__":
    
    # data for construnction of an ann.
    learning_rate = 1e-2
    inNumDim = 1
    outNumDim = 2
    numNodeList = [inNumDim,20,outNumDim]

    # sample data
    numSample = 1000
    t = np.linspace(0.1,1.5*np.pi,numSample)
    x = 3*np.cos(t)
    y = 1*np.sin(t)
    z = np.zeros([numSample,2]); z[:,0] = x[:]; z[:,1] = y[:]
    t = t.reshape([numSample,1])

    # create ann instance
    reg = ANN_Regression(learning_rate, numNodeList, active = tf.nn.softmax, prob_dropout = 1.0)

    # insert data
    reg.insert_data(t,z)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # train ann
    reg.run(sess,2000,0.8)

    # prediction data
    z = reg.predict(sess, t)

    with open('machine_test.out','w') as f:
        for i,row in enumerate(z):
            f.write('%f %f\n'%(row[0],row[1]))
