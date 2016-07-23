################################################################################
#
# \file    TestSpatialTransformerLayer.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Friday July 22, 2016
# \brief   A python script to test the SpatialTransformerLayer functionality
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

from SpatialTransformerLayer import SpatialTransformerLayer

logger = logging.getLogger('TestSpatialTransformerLayer')

def assertEqual(left, right):
    assert np.isclose(left, right).all()

def assertRotated(left, right):
    logger.debug("Rotation Test")
    transposedRight = np.transpose(right, (1, 0, 2))
    logger.debug("Right")
    logger.debug(right)
    logger.debug("Result")
    logger.debug(left)
    logger.debug("Reference")
    logger.debug(transposedRight)
    assert np.isclose(left, transposedRight).all()

def runTest():
    imageSize = 4
    imageChannels = 1

    # create spatial transformer layer
    stLayer = SpatialTransformerLayer(imageSize, imageSize, imageChannels, imageSize, imageSize, imageChannels, "Unitary")
    logger.debug("Created new spatial transformer layer")

    # create a random image
    np.random.seed(1)
    image = np.random.rand(imageSize, imageSize, imageChannels)

    inputImage = tf.constant(image, dtype=tf.float32)
    
    session = tf.Session()

    identity = stLayer.forward(inputImage)
    result = session.run(identity)
    assertEqual(result, image)
    logger.info ('Identity Test Passed')

    rtLayer = SpatialTransformerLayer(imageSize, imageSize, imageChannels, imageSize, imageSize, imageChannels, "Rotary")
    rotated = rtLayer.forward(inputImage)
    result2 = session.run(rotated)
    assertRotated(result2, image)
    logger.info ('Rotation Test Passed')




def main():
    parser = argparse.ArgumentParser(description="Test spatial transformer layer")
    parser.add_argument("-v", "--verbose"     , default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose      = arguments['verbose']
    
    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

    runTest()


if __name__ == '__main__':
    main()

