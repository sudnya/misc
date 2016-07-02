###############################################################################
#
# \file    sweepEstimate.py
# \author  Sudnya Padalikar <sudnya@attune.co>
# \date    Friday July 1, 2016
# \brief   A python script to estimate performance of various configurations
#
###############################################################################

import os
import argparse
import logging
import json

logger = logging.getLogger('sweepEstimate')

def getTotalImageSize(dims):
    retVal = 1
    for i in dims:
        retVal *= i

    return retVal

def getInputSize(dims):
    return dims[0] * dims[1] * dims[2]

def getOutputSize(dims):
    return dims[0] * dims[1] * dims[3]

def getFilterSize(fil, imageSize):
    return fil*fil*imageSize[2]*imageSize[3]


def estimatePerformanceWithRooflineModel(imageSize, filterSize, gpuPerformance):
    imageBytes  = (getInputSize(imageSize) + getOutputSize(imageSize)) * gpuPerformance['word-size'] * gpuPerformance['mini-batch-size']
    filterBytes = getFilterSize(filterSize, imageSize) * gpuPerformance['word-size']
    
    totalBytes = imageBytes + filterBytes

    dataLoadAndStoreTime = totalBytes / (gpuPerformance['bandwidth'] * gpuPerformance['cudnn-memory-efficiency'])   

    totalFlops = gpuPerformance['mini-batch-size'] * getTotalImageSize(imageSize) * filterSize * filterSize * 2

    totalThreads = gpuPerformance['sms'] * gpuPerformance['datapaths-per-sm']

    flopsPerThread = totalFlops / totalThreads

    utilization = min(gpuPerformance['work-per-datapath'], flopsPerThread) / gpuPerformance['work-per-datapath']

    totalMathTime = totalFlops / (gpuPerformance['flops'] * utilization * gpuPerformance['cudnn-math-efficiency'])

    kernelTime = max(totalMathTime, dataLoadAndStoreTime)

    totalTime = kernelTime + gpuPerformance['kernel-overhead']

    logger.info("Image: " + str(imageSize) + ", Filter: " + str(filterSize) + ", mem time: " +
        str(dataLoadAndStoreTime * 1.0e6) + " us, math time: " + str(totalMathTime * 1.0e6) +
        " us, total time: " + str(totalTime * 1.0e6) + " us (" + str(totalFlops / (totalTime * 1.0e12)) + " TFLOP/s) (" + str(gpuPerformance['kernel-overhead'] * 100. / kernelTime) + "% sync overhead)")

    return totalTime

def generateEstimates():
    gpuPerformance = {
        "flops"     : 6.144e12,
        "bandwidth" : 336.e9,
        "sms" : 24,
        "datapaths-per-sm" : 128,
        "work-per-datapath" : 1000.,
        "kernel-overhead" : 0.0,
        "word-size" : 4,
        "cudnn-math-efficiency" : 0.5,
        "cudnn-memory-efficiency" : 0.25,
        "mini-batch-size" : 1
    }

    targetMiniBatchSize = 64.

    filterSizes = [1, 3, 5]
    #TODO: channelSizes = [64, 128, 256]
    imageSizes = [[256, 256, 3, 64], [128, 128, 64, 64], [64, 64, 64, 128], [32, 32, 128, 128], [16, 16, 128, 256], [8, 8, 256, 256], [4, 4, 256, 256]]

    for j in filterSizes:
        totalTime = 0.0
        print "For filter size " + str(j)
        for i in imageSizes:
            totalTime += estimatePerformanceWithRooflineModel(i, j, gpuPerformance)
        print " Total network time " + str(totalTime * targetMiniBatchSize * 1.0e6 / gpuPerformance['mini-batch-size']) + "us"

    


def main():
    parser = argparse.ArgumentParser(description="Performance Estimation tool")
    parser.add_argument("-v", "--verbose",        default = False, action = "store_true")
    
    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']
    
    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        generateEstimates()

    except ValueError as e:
        logger.error ("Invalid Arguments: " + str(e))
        logger.error (parser.print_help())

    except Exception as e:
        logger.error ("Configuration Failed: " + str(e))
    

if __name__ == '__main__':
    main()

