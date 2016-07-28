################################################################################
#
# \file    NonLinearityFactory.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Tuesday July 26, 2016
# \brief   Returns a ReLu 
#
################################################################################

import os

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"

import tensorflow as tf
import numpy as np
import logging

from ReluNonLinearity import ReluNonLinearity
from NullNonLinearity import NullNonLinearity

logger = logging.getLogger('NonLinearityFactory')

class NonLinearityFactory:
    def __init__(self):
        pass

    @staticmethod
    def create(name):
        retVal = None
        if name == "ReLu":
            logger.info ("Creating a ReLu")
            retVal = ReluNonLinearity()
        elif name == "Null":
            logger.info ("Creating no nonlinearity ")
            retVal = NullNonLinearity()


        return retVal

