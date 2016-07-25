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

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def upsample2x(right):
    w = right.shape[0]
    h = right.shape[1]
    c = right.shape[2]

    result = np.zeros([w,h,c])

    for indexW in range(w):
        for indexH in range(h):
            for indexC in range(c):
                inputW = indexW * 0.5
                inputH = indexH * 0.5
                inputC = indexC

                if indexW % 2 == 0:
                    floorW = max(0, inputW - 1)
                    ceilW  = min(w, inputW + 1)

                    floorWIndex = int(inputW)
                    ceilWIndex = int(inputW)
                else:
                    floorW = int(np.floor(inputW))
                    ceilW  = int(np.ceil(inputW))

                    floorWIndex = floorW
                    ceilWIndex = ceilW
                
                if indexH % 2 == 0:
                    floorH = max(0, inputH - 1)
                    ceilH  = min(h, inputH + 1)
                    
                    floorHIndex = int(inputH)
                    ceilHIndex = int(inputH)
                else:
                    floorH = int(np.floor(inputH))
                    ceilH  = int(np.ceil(inputH))
                    
                    floorHIndex = floorH
                    ceilHIndex = ceilH

                points = [(floorW, floorH, right[floorWIndex, floorHIndex, inputC]),
                          (floorW, ceilH,  right[floorWIndex, ceilHIndex,  inputC]),
                          (ceilW,  floorH, right[ceilWIndex,  floorHIndex, inputC]),
                          (ceilW,  ceilH,  right[ceilWIndex,  ceilHIndex,  inputC])]

                result[indexW, indexH, indexC] = bilinear_interpolation(inputW, inputH, points)

    return result
    
def assertScaled(left, right):
    logger.debug("Scaled Test")
    scaledRight = upsample2x(right)
    logger.debug("Right")
    logger.debug(right)
    logger.debug("Result")
    logger.debug(left)
    logger.debug("Reference")
    logger.debug(scaledRight)
    assert np.isclose(left, scaledRight).all()


def assertManualWithOffset(left, right):
    logger.debug("Scaled with offset Test")

    #reference = np.empty_list(right)
    reference = np.array([3.5 + i * 0.5 for i in range(16)], dtype=np.float)
    reference = np.reshape(reference, right.shape)


    logger.debug("Result")
    logger.debug(left)
    logger.debug("Reference")
    logger.debug(reference)
    assert np.isclose(left, reference).all()


def runTest():
    imageSize = 4
    imageChannels = 1

    # create spatial transformer layer
    logger.debug("Created new spatial transformer layer")

    # create a random image
    np.random.seed(1)
    image = np.random.rand(imageSize, imageSize, imageChannels)

    inputImage = tf.constant(image, dtype=tf.float32)
    
    session = tf.Session()

    identityLayer = SpatialTransformerLayer(imageSize, imageSize, imageChannels, imageSize, imageSize, imageChannels, "Unitary")
    identity      = identityLayer.forward(inputImage)
    result1       = session.run(identity)
    
    assertEqual(result1, image)
    logger.info ('Identity Test Passed')

    rotatingLayer = SpatialTransformerLayer(imageSize, imageSize, imageChannels, imageSize, imageSize, imageChannels, "Rotary")
    rotated       = rotatingLayer.forward(inputImage)
    result2       = session.run(rotated)
    
    assertRotated(result2, image)
    logger.info ('Rotation Test Passed')

    scalingLayer = SpatialTransformerLayer(imageSize, imageSize, imageChannels, imageSize, imageSize, imageChannels, "Scaled")
    scaled = scalingLayer.forward(inputImage)
    result3 = session.run(scaled)
    assertScaled(result3, image)
    logger.info ('Scaling Test Passed')

    scalingWithOffsetLayer = SpatialTransformerLayer(imageSize, imageSize, imageChannels, imageSize, imageSize, imageChannels, "ScaledWithOffset")
    manual = np.arange(1, (imageSize*imageSize + 1), dtype=np.float)
    manual = np.reshape(manual, (imageSize, imageSize, imageChannels))
    manualImage = tf.constant(manual, dtype=tf.float32)
    scaledWithOffset = scalingWithOffsetLayer.forward(manualImage)
    result4 = session.run(scaledWithOffset)
    assertManualWithOffset(result4, manual)
    logger.info ('Scaling with offset Test Passed')



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

