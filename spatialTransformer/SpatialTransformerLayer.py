################################################################################
#
# \file    TestSpatialTransformerLayer.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Friday July 22, 2016
# \brief   A python script to implement the SpatialTransformerLayer 
#          functionality
#
################################################################################

import os

os.environ["LD_LIBRARY_PATH"]= "/usr/local/cuda/lib"
import tensorflow as tf
import math

from RotaryLayer  import RotaryLayer
from ScaledLayer  import ScaledLayer
from UnitaryLayer import UnitaryLayer
from ScaledWithOffsetLayer  import ScaledWithOffsetLayer
from FullyConnectedLayer import FullyConnectedLayer
from ConvLayer import ConvLayer
from NeuralNetwork import NeuralNetwork

class SpatialTransformerLayer:
    def __init__(self, inputW, inputH, inputC, outputW, outputH, outputC, locType):
        self.inputW = inputW
        self.inputH = inputH
        self.inputC = inputC

        self.outputW = outputW
        self.outputH = outputH
        self.outputC = outputC

        self.localizationType = locType

        self.localizationNetwork = self.createLocalizationNetwork()

    def initialize(self):
        self.localizationNetwork.initialize()

    def createLocalizationNetwork(self):
        if self.localizationType == "Rotary":
            return RotaryLayer()
        if self.localizationType == "Scaled":
            return ScaledLayer()
        if self.localizationType == "ScaledWithOffset":
            return ScaledWithOffsetLayer()
        if self.localizationType == "Unitary":
            return UnitaryLayer()
        if self.localizationType == "FullyConnected":
            network = NeuralNetwork()

            network.addLayer(FullyConnectedLayer(self.inputW * self.inputH * self.inputC, 32, 0, "ReLu"))
            network.addLayer(FullyConnectedLayer(32, 3*4, 1, "ReLu"))

            return network
        if self.localizationType == "ConvLayer":
            network = NeuralNetwork()
            
            network.addLayer(ConvLayer((self.inputW, self.inputH, self.inputC), (3, 3, self.inputC, self.inputC), 0, "ReLu"))
            network.addLayer(FullyConnectedLayer(self.inputW * self.inputH * self.inputC, 3*4, 1, "ReLu"))

            return network

    def clampToInputBoundary(self, transformedCoordinates):

        sliceW = tf.slice(transformedCoordinates, [0, 2], [tf.shape(transformedCoordinates)[0], 1])

        sliceW = tf.maximum(sliceW, tf.constant([0], dtype=tf.float32))
        sliceW = tf.minimum(sliceW, tf.constant([self.inputW], dtype=tf.float32))
        
        sliceH = tf.slice(transformedCoordinates, [0, 1], [tf.shape(transformedCoordinates)[0], 1])

        sliceH = tf.maximum(sliceH, tf.constant([0], dtype=tf.float32))
        sliceH = tf.minimum(sliceH, tf.constant([self.inputH], dtype=tf.float32))

        sliceC = tf.slice(transformedCoordinates, [0, 0], [tf.shape(transformedCoordinates)[0], 1])

        sliceC = tf.maximum(sliceC, tf.constant([0], dtype=tf.float32))
        sliceC = tf.minimum(sliceC, tf.constant([self.inputC], dtype=tf.float32))

        return tf.concat(1, [sliceC, sliceH, sliceW])

    def forward(self, inputData):
        #inputData = tf.Print(inputData, [inputData], message= "Input", summarize=100)

        #(1). localisation
        #(2). transform with theta
        theta = tf.transpose(tf.reshape(self.localizationNetwork.forward(inputData), [-1, 3, 4]), perm=[2, 0, 1])
        
        #theta = tf.Print(theta, [theta], message= "Theta", summarize=100)

        #(3). get coordinates in matrix unrolled
        coordinatesMatrix = self.getCoordinates()
        #coordinatesMatrix = tf.Print(coordinatesMatrix, [coordinatesMatrix], message= "coordinatesMatrix", summarize=100)

        #(4). dot product of (3) and (2)
        transformedCoordinates = tf.reshape(tf.matmul(coordinatesMatrix, tf.reshape(theta, [-1, 3 * tf.shape(inputData)[0]])),
                                            [-1, tf.shape(theta)[1], 3])
        transformedCoordinates = tf.reshape(tf.transpose(transformedCoordinates, [1, 0, 2]), [-1, 3])

       # transformedCoordinates = tf.Print(transformedCoordinates, [transformedCoordinates], message= "transformedCoordinates", summarize=100)

        transformedCoordinates = self.clampToInputBoundary(transformedCoordinates)

        #(5). bi-linear sampling at input matrix where coordinates are transformed from step (4)
        outputMatrix = self.bilinear(inputData, transformedCoordinates)

        #(6). Step (5) is output matrix --> reshape
        result = tf.reshape(outputMatrix, [-1, self.outputC, self.outputH, self.outputW])
        
        #result = tf.Print(result, [result], message= "Result", summarize=100)

        return result
        

    #def backPropData(self, outputDeltas):
    #    pass

    #def backPropGradients(self, outputDeltas):
    #    pass


    def getWeights(self):
        pass

    def getCoordinates(self):
        retVal = []
        for i in range(0, self.outputC):
            for j in range(0, self.outputH):
                for k in range(0, self.outputW):
                    retVal.append([i, j, k, 1.0])

        return tf.constant(retVal, dtype=tf.float32)


    def getSampledPositions(self, transformedCoordinates):
        options = [
                    (0,0,0) ,
                    (0,0,1) ,
                    (0,1,0) ,
                    (0,1,1) ,
                    (1,0,0) ,
                    (1,0,1) ,
                    (1,1,0) ,
                    (1,1,1)
                ]

        batchIds = self.getBatchIds(transformedCoordinates)

        result = []

        for ceilW, ceilH, ceilC in options:
            w = self.clampSlice(ceilW, transformedCoordinates, 2)
            h = self.clampSlice(ceilH, transformedCoordinates, 1)
            c = self.clampSlice(ceilC, transformedCoordinates, 0)
            
            result.append(tf.concat(1, [batchIds, c, h, w]))

        return result

    def getBatchIds(self, transformedCoordinates):
        batchId = tf.zeros([tf.shape(transformedCoordinates)[0]], dtype=tf.int32)
        batchId = tf.reshape(batchId, [-1, self.outputC * self.outputH * self.outputW])
        
        linearIds = tf.reshape(tf.range(tf.shape(batchId)[0]), [tf.shape(batchId)[0], 1])
        batchId = tf.reshape(tf.add(linearIds, batchId), [tf.shape(transformedCoordinates)[0], 1])

        return tf.cast(batchId, tf.float32)

    def clampSlice(self, shouldCeil, transformedCoordinates, index):
        coordinateSlice = tf.slice(transformedCoordinates, [0, index], [tf.shape(transformedCoordinates)[0], 1])

        if not shouldCeil:
            result = tf.floor(coordinateSlice)
        else:
            result = tf.ceil(coordinateSlice)

        return result

    def sliceIndex(self, left, index):
        return tf.reshape(tf.slice(left, [0, index], [tf.shape(left)[0], 1]), [tf.shape(left)[0]])

    def getNeighborWeights(self, transformedCoordinates, clampedCoordinatesList):
        flooredCoordinates = tf.slice(clampedCoordinatesList[0], [0, 1], [tf.shape(clampedCoordinatesList[0])[0], 3])

        #transformedCoordinates = tf.Print(transformedCoordinates, [transformedCoordinates], summarize=1000)
        #flooredCoordinates = tf.Print(flooredCoordinates, [flooredCoordinates], summarize=1000)
        deltas = tf.sub(transformedCoordinates, flooredCoordinates)
        #deltas = tf.Print(deltas, [deltas], summarize=1000)

        deltaW = self.sliceIndex(deltas, 2)
        deltaH = self.sliceIndex(deltas, 1)
        deltaC = self.sliceIndex(deltas, 0)

        #deltaW = tf.Print(deltaW, [deltaW], summarize=1000)
        #deltaH = tf.Print(deltaH, [deltaH], summarize=1000)
        #deltaC = tf.Print(deltaC, [deltaC], summarize=1000)

        #just declare for concisely writing the various weights
        ConstantOne = tf.constant([1], dtype=tf.float32)

        W_lll = tf.mul(tf.mul(tf.sub(ConstantOne, deltaW) , tf.sub(ConstantOne, deltaH)) , tf.sub(ConstantOne, deltaC))
        W_llu = tf.mul(tf.mul(tf.sub(ConstantOne, deltaW) , tf.sub(ConstantOne, deltaH)) , deltaC                     )
        W_lul = tf.mul(tf.mul(tf.sub(ConstantOne, deltaW) , deltaH                     ) , tf.sub(ConstantOne, deltaC))
        W_luu = tf.mul(tf.mul(tf.sub(ConstantOne, deltaW) , deltaH                     ) , deltaC                     )
        W_ull = tf.mul(tf.mul(deltaW                      , tf.sub(ConstantOne, deltaH)) , tf.sub(ConstantOne, deltaC))
        W_ulu = tf.mul(tf.mul(deltaW                      , tf.sub(ConstantOne, deltaH)) , deltaC                     )
        W_uul = tf.mul(tf.mul(deltaW                      , deltaH                     ) , tf.sub(ConstantOne, deltaC))
        W_uuu = tf.mul(tf.mul(deltaW                      , deltaH                     ) , deltaC                     )

        #W_lll = tf.Print(W_lll, [W_llu], summarize=1000)
        #W_llu = tf.Print(W_llu, [W_lll], summarize=1000)
        #W_lul = tf.Print(W_lul, [W_lul], summarize=1000)
        #W_luu = tf.Print(W_luu, [W_luu], summarize=1000)
        #W_ull = tf.Print(W_ull, [W_ull], summarize=1000)
        #W_ulu = tf.Print(W_ulu, [W_ulu], summarize=1000)
        #W_uul = tf.Print(W_uul, [W_uul], summarize=1000)
        #W_uuu = tf.Print(W_uuu, [W_uuu], summarize=1000)

        weightList = []

        weightList.append(W_lll) 
        weightList.append(W_llu) 
        weightList.append(W_lul) 
        weightList.append(W_luu) 
        weightList.append(W_ull) 
        weightList.append(W_ulu) 
        weightList.append(W_uul) 
        weightList.append(W_uuu) 
       

        return weightList

    def gatherData(self, inputData, sampledPositions):

        result = []

        flatInputData = tf.reshape(inputData, [-1])

        inputShape = tf.shape(inputData)

        inputStrides = tf.constant([[self.inputW * self.inputH * self.inputC], [self.inputW * self.inputH], [self.inputW], [1]],
                dtype=tf.float32)

        for positions in sampledPositions:

            flatPositions = tf.reshape(tf.matmul(positions, inputStrides), [-1])

            gatheredData = tf.gather(flatInputData, tf.to_int64(flatPositions))

            result.append(gatheredData)

        return result

    def bilinear(self, inputData, transformedCoordinates):
        # (1). for each input dimension get all nearest neighbors -> 2^n for n dims
        sampledPositions = self.getSampledPositions(transformedCoordinates)

        # (2). get a weight per value in above step
        weightedDistance = self.getNeighborWeights(transformedCoordinates, sampledPositions)
        
        # (3). get inputData value at that location
        data = self.gatherData(inputData, sampledPositions)
        
        # (4). multiply weight with value
        accumulator = tf.zeros(tf.shape(data[0]), dtype=tf.float32)

        for value, weight in zip(data, weightedDistance):
            accumulator = tf.add(tf.mul(value, weight), accumulator)

        #accumulator = tf.Print(accumulator, [accumulator], summarize=1000)
        return accumulator

