################################################################################
#
# \file    ReluNonLinearity.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Tuesday July 26, 2016
# \brief   ReLu class 
#
################################################################################

import os

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"

import tensorflow as tf
import numpy as np

class ReluNonLinearity:
    def __init__(self):
        pass

    def forward(self, inputMat):
        return tf.nn.relu(inputMat)


