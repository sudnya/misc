################################################################################
#
# \file    ScaledUpLayer.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Friday August 5, 2016
# \brief   Returns scaled up tensor for theta (required for a test case)
#
################################################################################

import os

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"

import tensorflow as tf
import numpy as np

class ScaledUpLayer:
    def __init__(self):
        pass

    def forward(self, inputData):
        theta  = [[   # c_in  h_in  w_in  offset
                    [1.0,   0.0,  0.0,  0.0], # c_out
                    [0.0,   2.0,  0.0,  0.0], # h_out
                    [0.0,   0.0,  2.0,  0.0]  # w_out
                 ]]
        theta = tf.constant(theta, dtype=tf.float32)

        zeros = tf.zeros([tf.shape(inputData)[0], 3, 4], dtype=tf.float32)

        theta = tf.add(theta, zeros);

        return theta


