################################################################################
#
# \file    NeuralNetworkBuilder.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Tuesday July 26, 2016
# \brief   A factory to created various types of NN
#
################################################################################

import logging

from ConvLayer import ConvLayer
from FullyConnectedLayer import FullyConnectedLayer
from SpatialTransformerLayer import SpatialTransformerLayer

from NeuralNetwork import NeuralNetwork

logger = logging.getLogger('NeuralNetworkBuilder')

class NeuralNetworkBuilder:
    def __init__(self):
        pass

    @staticmethod
    def createFullyConnectedNetwork(parameters):
        logger.info ("Creating a fully connected network")
        network = NeuralNetwork()
        
        idx = 0
        for inputSize, outputSize in parameters:
            isLastLayer = (idx == (len(parameters) - 1))

            if isLastLayer:
                nonlinearity = "Null"
            else:
                nonlinearity = "ReLu"

            network.addLayer(FullyConnectedLayer(inputSize, outputSize, idx, nonlinearity))
            idx += 1

        return network

    @staticmethod
    def createConvNetwork(parameters):
        logger.info ("Creating a convolutional network")
        network = NeuralNetwork()
        
        idx = 0
        for inputSize, outputSize in parameters:
            isLastLayer = (idx == (len(parameters) - 1))

            if isLastLayer:
                nonlinearity = "Null"
                network.addLayer(FullyConnectedLayer(inputSize, outputSize, idx, nonlinearity))
            else:
                nonlinearity = "ReLu"
                network.addLayer(ConvLayer(inputSize, outputSize, idx, nonlinearity))

            idx += 1

        return network

    @staticmethod
    def createSpatialTransformerWithFullyConnectedNetwork(parameters):
        logger.info ("Creating a fully connected network with a spatial transformer input layer")
        network = NeuralNetwork()
        
        idx = 0
        for inputSize, outputSize in parameters:
            isLastLayer = (idx == (len(parameters) - 1))

            if isLastLayer:
                nonlinearity = "Null"
            else:
                nonlinearity = "ReLu"

            if idx == 0:
                network.addLayer(SpatialTransformerLayer(inputSize[0], inputSize[1], inputSize[2],
                    outputSize[0], outputSize[1], outputSize[2], "FullyConnected"))
            else:
                network.addLayer(FullyConnectedLayer(inputSize, outputSize, idx, nonlinearity))
            idx += 1

        return network

    @staticmethod
    def createSpatialTransformerWithConvNetwork(parameters):
        return None




