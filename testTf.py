###############################################################################
#
# \file    testTf.py
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
import tensorflow as tf

logger = logging.getLogger('testTf')

def benchMark(session):
    # Create a Constant op that produces a 1x2 matrix.  The op is
    # added as a node to the default graph.
    #
    # The value returned by the constructor represents the output
    # of the Constant op.
    matrix1 = tf.constant([[3., 3.]])

    # Create another Constant that produces a 2x1 matrix.
    matrix2 = tf.constant([[2.],[2.]])

    # Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
    # The returned value, 'product', represents the result of the matrix
    # multiplication.
    product = tf.matmul(matrix1, matrix2)


    result = session.run(product)

    #print(result)


def runTf():
    start = time.time()
    iterations = 1000
    
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
        runTf()

    except ValueError as e:
        logger.error ("Invalid Arguments: " + str(e))
        logger.error (parser.print_help())

    except Exception as e:
        logger.error ("Configuration Failed: " + str(e))
    

if __name__ == '__main__':
    main()


