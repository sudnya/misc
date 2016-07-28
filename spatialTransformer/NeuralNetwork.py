################################################################################
#
# \file    NeuralNetwork.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Tuesday July 26, 2016
# \brief   Returns a configured NN
#
################################################################################

import os

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"

import tensorflow as tf


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def initialize(self):
        for layer in self.layers:
            layer.initialize()

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        result = inputs

        for layer in self.layers:
            result = layer.forward(result)

        return result




