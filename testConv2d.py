###############################################################################
#
# \file    testConv2d.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Sudnya July 3, 2016
# \brief   A python script to try out tensor flow
#
###############################################################################

import os
import argparse
import logging
import json
import time

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"
import tensorflow as tf

logger = logging.getLogger('testConv2d')

class ConvOptions:
    def __init__(self, imageW, imageH, imageChannels, miniBatchSize, filterW, filterH, filterCount, iterations):
        self.imageW = imageW;
        self.imageH = imageH;
        self.imageChannels = imageChannels;
        self.miniBatchSize = miniBatchSize;

        self.filterW = filterW;
        self.filterH = filterH;
        self.filterCount = filterCount;

        self.iterations = iterations;

def benchMark(session):
    miniBatchSize = 1
    iterations = 1000
    strides = [1,1,1,1]
    padding = "SAME"

    # input, filter, output matrices
    options  = ConvOptions(256, 256, 3, miniBatchSize, 3, 3, 64, iterations);
    inputM   = tf.zeros([miniBatchSize, options.imageH, options.imageW, options.imageChannels], dtype=tf.float32)
    filterM  = tf.zeros([options.filterH, options.filterW, options.imageChannels, options.filterCount], dtype=tf.float32)
    outputM  = tf.zeros([miniBatchSize, options.imageH, options.imageW, options.filterCount], dtype=tf.float32)

    basicOp = tf.nn.conv2d(inputM, filterM, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)

    result = session.run(basicOp)

    #print(result)


def runConvolution():
    start = time.time()
    iterations = 1
    
    session = tf.Session()
    for i in range(iterations):
        benchMark(session)

    session.close()

    end = time.time()

    seconds = end - start

    print "Time per call is " + str(seconds * 1.e6 / iterations) + "us"
    

def main():
    parser = argparse.ArgumentParser(description="Tensorflow test tool")
    parser.add_argument("-v", "--verbose",        default = False, action = "store_true")
    
    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']
    
    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        runConvolution()

    except ValueError as e:
        logger.error ("Invalid Arguments: " + str(e))
        logger.error (parser.print_help())

    except Exception as e:
        logger.error ("Configuration Failed: " + str(e))
    

if __name__ == '__main__':
    main()


