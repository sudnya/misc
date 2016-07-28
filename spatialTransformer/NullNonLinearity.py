################################################################################
#
# \file    NullNonLinearity.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Tuesday July 26, 2016
# \brief   Null class 
#
################################################################################

import os

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"

import tensorflow as tf
import numpy as np

class NullNonLinearity:
    def __init__(self):
        pass

    def forward(self, inputMat):
        return inputMat



