################################################################################
#
# \file    UnitaryLayer.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Friday July 22, 2016
# \brief   Returns unit tensor for theta
#
################################################################################

import os

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"

import tensorflow as tf
import numpy as np

class UnitaryLayer:
    def __init__(self):
        pass

    def forward(self, inputData):
        theta = [[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]
                ]]

        theta = tf.constant(theta, dtype=tf.float32)

        zeros = tf.zeros([tf.shape(inputData)[0], 3, 4], dtype=tf.float32)

        theta = tf.add(theta, zeros);

        return theta

    def initialize(self):
        pass


