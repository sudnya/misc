################################################################################
#
# \file    ScaledLayer.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Saturday July 23, 2016
# \brief   Returns scaled tensor for theta
#
################################################################################

import os

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"

import tensorflow as tf
import numpy as np

class ScaledLayer:
    def __init__(self):
        pass

    def forward(self, inputData):
        theta  = [   # c_in  h_in  w_in  offset
                    [1.0,   0.0,  0.0,  0.0], # c_out
                    [0.0,   0.5,  0.0,  0.0], # h_out
                    [0.0,   0.0,  0.5,  0.0]  # w_out
                 ]
        return tf.constant(theta)


