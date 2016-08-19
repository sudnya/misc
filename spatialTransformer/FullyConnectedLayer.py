################################################################################
#
# \file    FullyConnectedLayer.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Tuesday July 26, 2016
# \brief   Returns a fully connected layer
#
################################################################################

import os

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"

import tensorflow as tf
import numpy as np
from NonLinearityFactory import NonLinearityFactory
import math

class FullyConnectedLayer:
    def __init__(self, inputSize, outputSize, idx, nonlinearityType):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.idx = idx
        self.nonlinearityType = nonlinearityType

    def initialize(self):
        with tf.name_scope('FullyConnected' + str(self.idx) ):
            self.W = self.createWeightMatrix(self.inputSize, self.outputSize)
            self.b = self.createBiasMatrix(self.outputSize)

        self.nonlinearity = NonLinearityFactory.create(self.nonlinearityType)
        self.weights = [self.W, self.b]


    def forward(self, inputData):
        result = tf.reshape(inputData, [-1, self.inputSize])
        result = tf.matmul(result, self.W)
        result = tf.add(self.b, result)

        return self.nonlinearity.forward(result)


    def getWeights(self):
        return self.weights

    def createWeightMatrix(self, inputSize, outputSize):
        return tf.Variable( tf.truncated_normal([inputSize, outputSize], stddev=math.sqrt(6.0 / float(inputSize+outputSize)), seed=1), name='weights')
    
    def createBiasMatrix(self, outputSize):
        return tf.Variable(tf.zeros([outputSize]), name='biases')

