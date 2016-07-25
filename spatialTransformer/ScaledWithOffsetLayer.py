################################################################################
#
# \file    ScaledWithOffsetLayer.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Saturday July 23, 2016
# \brief   Returns scaled tensor for theta
#
################################################################################

import os

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"

import tensorflow as tf
import numpy as np

class ScaledWithOffsetLayer:
    def __init__(self):
        pass

    def forward(self, inputData):
        theta  = [   # w_in  h_in  c_in  offset
                    [0.5,   0.0,  0.0,  0.5], # w_out
                    [0.0,   0.5,  0.0,  0.5], # h_out
                    [0.0,   0.0,  1.0,  0.0]  # c_out
                 ]
        return tf.constant(theta)


