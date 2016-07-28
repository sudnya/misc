################################################################################
#
# \file    ConvLayer.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Tuesday July 26, 2016
# \brief   Returns a convolutional layer
#
################################################################################

import math
import os

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"

import tensorflow as tf
import numpy as np

from NonLinearityFactory import NonLinearityFactory

class ConvLayer:
    def __init__(self, inputSize, filterSize, idx, nonlinearityType):
        self.strides = [1,1,1,1]
        self.padding = "SAME"
        self.dataFormat = "NCHW"

        self.inputSize = inputSize
        self.filterSize = filterSize
        self.idx = idx
        self.nonlinearityType = nonlinearityType
        
        self.nonlinearity = NonLinearityFactory.create(self.nonlinearityType)

    def initialize(self):

        self.filterM = self.createFilterMatrix()
        self.filterB = self.createBiasMatrix()

        self.weights = [self.filterM, self.filterB]

        

    def forward(self, inputData):
        inputData = tf.reshape(inputData, [-1, self.inputSize[2], self.inputSize[1], self.inputSize[0]])
        result = tf.nn.conv2d(inputData, self.filterM, self.strides, self.padding, use_cudnn_on_gpu=None, data_format=self.dataFormat, name=None)
        result = tf.add(result, self.filterB)

        return self.nonlinearity.forward(result)

    def getWeights(self):
        return self.weights

    def createFilterMatrix(self):
        filterW = self.filterSize[0]
        filterH = self.filterSize[1]
        inputChannels = self.inputSize[2]
        outputChannels = self.filterSize[2]

        return tf.Variable(tf.truncated_normal([filterH, filterW, inputChannels, outputChannels],
            stddev=1.0 / math.sqrt(float(filterW*filterH*inputChannels)), seed=1), name='weights')
    
    def createBiasMatrix(self):

        outputChannels = self.filterSize[2]

        return tf.Variable(tf.zeros([outputChannels, 1, 1]), name='biases')

