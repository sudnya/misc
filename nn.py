###############################################################################
#
# \file    nn.py
# \author  Sudnya Diamos <mailsudnya@gmail.com>
# \date    Thursday Jan 26, 2017
# \brief   A python script to try out handwritten NN concepts
#
###############################################################################

import os
import argparse
import logging
import json
import time
import numpy as np
import math

logger = logging.getLogger('myNN')
INPUTLAYERSIZE = 3
OUTPUTLAYERSIZE = 3
SAMPLECOUNT = 2

# X -> (*) -> p0 -> (+) -> p1 -> (~) -> yHat
#       W1           B1        

class NeuralNetwork():
    def __init__(self, nonLin, loss, inputLayerSize, outputLayerSize):
        self.nonLinearity = nonLin
        self.lossFunction = loss
        self.W1           = np.random.randn(outputLayerSize, inputLayerSize)
        self.B1           = np.zeros([outputLayerSize, 1])

    def forward(self, X):
        self.X  = X
        self.p0 = np.dot(self.W1, self.X)
        self.p1 = self.p0 + self.B1
        yHat    = self.__applyNonLinearity__(self.p1)
        
        logger.info("Input X: \n" + str(self.X.shape))
        logger.info("Pre-activations: \n" + str(self.p0.shape))
        logger.info("With Bias: \n" + str(self.p1.shape))
        logger.info("Output: \n" + str(yHat.shape))
        
        logger.debug("Input X: \n" + str(self.X))
        logger.debug("Pre-activations: \n" + str(self.p0))
        logger.debug("With Bias: \n" + str(self.p1))
        logger.debug("Output: \n" + str(yHat))
        
        return yHat


    def backProp(self, y, yPredicted):
        # get dL/dyPredicted
        dLdyPredicted = self.__costFunctionDelta__(y, yPredicted)

        # get gradients in all hidden layers #TODO: in a loop for stack of hidden layers
        dLdp1 = self.__applyNonLinearityDelta__(yPredicted) * dLdyPredicted

        # gradients just propagate back on add operation
        dLdp0 = dLdp1
        # bias is always just a vector regardless of minibatch size
        self.dLdB1 = np.reshape(np.add.reduce(dLdp1, 1), [dLdp1.shape[0],1])

        # transpose to line up matrix dimensions for compatible dot product
        self.dLdW1 = dLdp0.dot(np.transpose(self.X))
        
        logger.info("y: \n" + str(y.shape))
        logger.info("y predicted: \n" + str(yPredicted.shape))
        logger.info("dLdyPredicted : \n" + str(dLdyPredicted.shape))
        logger.info("dLdp1: \n" + str(dLdp1.shape))
        logger.info("dLdW1: \n" + str(self.dLdW1.shape))

        logger.debug("y: \n" + str(y))
        logger.debug("y predicted: \n" + str(yPredicted))
        logger.debug("dLdyPredicted : \n" + str(dLdyPredicted))
        logger.debug("dLdp1: \n" + str(dLdp1))
        logger.debug("dLdW1: \n" + str(self.dLdW1))

        return self.dLdW1


    def getWeights(self):
        return [self.W1, self.B1]
    
    def getGradients(self):
        return [self.dLdW1, self.dLdB1]

    def __applyNonLinearity__(self, z):
        if self.nonLinearity.lower() == "sigmoid":
            return self.__sigmoid__(z)
        else:
            assert false, "non linearity not supported"
            return -1


    def __applyNonLinearityDelta__(self, z):
        if self.nonLinearity.lower() == "sigmoid":
            return self.__sigmoidDelta__(z)
        else:
            assert false, "non linearity backprop not supported"
            return -1


    def __sigmoid__(self, z):
        return 1/(1+np.exp(-z))


    def __sigmoidDelta__(self, output):
        return output*(1-output)


    def costFunction(self, y, yHat):
        if self.lossFunction.lower() == "simple":
            return self.__simpleLoss__(y, yHat)
        else:
            assert false, "loss function not supported"
            return -1


    def __costFunctionDelta__(self, y, yHat):
        if self.lossFunction.lower() == "simple":
            return self.__simpleLossDelta__(y, yHat)
        else:
            assert false, "loss function backprop not supported"
            return -1


    def __simpleLoss__(self, y, yHat):
        return (0.5*np.sum((y-yHat)**2))/y.shape[1]

    def __simpleLossDelta__(self, y, yHat):
        return (yHat - y)/(y.shape[1])

class gradChecker():
    def __init__(self):
        # for each weight get cost (w + epsilon) and cost (w + epsilon)
        # subtract above and / 2 epsilon
        # compare with gradient at that location
        pass

    def run(self, X, y, nn, epsilon):
        retVal = True
        gradients = nn.getGradients()

        weights = nn.getWeights()  
        for w in range(len(weights)):
            shape = weights[w].shape
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    #logger.info("Weights:   \n" + str(nn.getWeights()))
                    
                    weights[w][i][j] += epsilon

                    #logger.info("New Weights:   \n" + str(nn.getWeights()))
                    predictionsP  = nn.forward(X)
                    plusCost      = nn.costFunction(y, predictionsP)
                    
                    weights[w][i][j] -= 2*epsilon

                    #logger.info("New Weights:   \n" + str(nn.getWeights()))
                    predictionsN  = nn.forward(X)
                    minusCost     = nn.costFunction(y, predictionsN)
                    
                    weights[w][i][j] += epsilon

                    computedGradient = (plusCost - minusCost) / (2*epsilon)

                    print "grad ", str(gradients[0].shape)
                    difference = np.power(computedGradient - gradients[w][i][j], 2) / np.power(gradients[w][i][j], 2)
                    
                    if difference > epsilon:
                        retVal = False
                        logger.info("Difference " + str(difference) + " between computed gradient " +
                                    str(computedGradient) + " and back prop gradient " + str(gradients[w][i][j]) +
                                    " not in acceptable range")
                    else:
                        logger.debug("Difference " + str(difference) + " between computed gradient " +
                                    str(computedGradient) + " and back prop gradient " + str(gradients[w][i][j]) +
                                    " is in acceptable range")
        
        return retVal


def main():
    # 1. support multiple mini batches in X - DONE
    # TODO: 
    # 2. optimizer - gradient descent
    # 3. more cost functions (softmax + cross entropy\negative log likelyhood)
    # 4. support ReLu
    #    more layers
    # 5. some toy dataset
    parser = argparse.ArgumentParser(description="My NN examples")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true")
    
    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']
    
    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    np.random.seed(23)
    myNN       = NeuralNetwork("sigmoid", "simple", INPUTLAYERSIZE, OUTPUTLAYERSIZE)
    X          = np.random.rand(INPUTLAYERSIZE, SAMPLECOUNT)
    y          = np.random.rand(OUTPUTLAYERSIZE, SAMPLECOUNT)
    prediction = myNN.forward(X)
    gradients  = myNN.backProp(y, prediction)
    
    gradCheck  = gradChecker()
    if (gradCheck.run(X, y, myNN, 1e-06)):
        print "Grad check passed"



    

if __name__ == '__main__':
    main()


