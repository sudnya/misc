################################################################################
#
# \file    RunMNIST.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Monday July 25, 2016
# \brief   A python script to train and test MNIST using Spatial Transformer 
#          Layer
#
################################################################################

import os
import json
import traceback
import argparse
import logging

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

from SpatialTransformerLayer import SpatialTransformerLayer

logger = logging.getLogger('RunMNIST')

def assertEqual(left, right):
    assert np.isclose(left, right).all()

def createNetwork(self, name, imageSize):
    layerList = []
    if name == "fullyConnected":
        l1 = FullyConnectedLayer(imageSize, imageSize)
        hidden1 = l1.forward(inputs)
        
        l2 = FullyConnectedLayer(imageSize, imageSize)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, l2.getWeights()) + l2.getBiases())

    
    

def runMNIST():
    imageSize = 4
    imageChannels = 1

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    self.createNetwork("fullyConnected", imageSize)
    # create spatial transformer layer
#    logger.debug("Created new spatial transformer layer")
#
#    # create a random image
#    np.random.seed(1)
#    image = np.random.rand(imageSize, imageSize, imageChannels)
#
#    inputImage = tf.constant(image, dtype=tf.float32)
#    
#    session = tf.Session()
#
#    identityLayer = SpatialTransformerLayer(imageSize, imageSize, imageChannels, imageSize, imageSize, imageChannels, "Unitary")
#    identity      = identityLayer.forward(inputImage)
#    result1       = session.run(identity)
#    
#    assertEqual(result1, image)
#    logger.info ('Identity Test Passed')

def main():
    parser = argparse.ArgumentParser(description="Test MNIST")
    parser.add_argument("-v", "--verbose"     , default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose      = arguments['verbose']
    
    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

    runMNIST()


if __name__ == '__main__':
    main()


