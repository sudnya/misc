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

def estimateLocalizationNetworkPerformanceWithRooflineModel(imageSize, filterSize, layers, gpuPerformance):
    imageBytes  = (getInputSize(imageSize) + getOutputSize(imageSize)) * gpuPerformance['word-size'] * gpuPerformance['mini-batch-size']
    filterBytes = getFilterSize(filterSize, imageSize) * gpuPerformance['word-size']

    totalBytes = layers * (imageBytes + filterBytes)

    dataLoadAndStoreTime = totalBytes / (gpuPerformance['bandwidth'] * gpuPerformance['cudnn-memory-efficiency'])

    totalFlops = gpuPerformance['mini-batch-size'] * getTotalImageSize(imageSize) * filterSize * filterSize * 2 * layers

    totalThreads = gpuPerformance['sms'] * gpuPerformance['datapaths-per-sm']

    flopsPerThread = totalFlops / totalThreads

    utilization = min(gpuPerformance['work-per-datapath'], flopsPerThread) / gpuPerformance['work-per-datapath']

    totalMathTime = totalFlops / (gpuPerformance['flops'] * utilization * gpuPerformance['cudnn-math-efficiency'])

    kernelTime = max(totalMathTime, dataLoadAndStoreTime)

    totalTime = kernelTime + gpuPerformance['kernel-overhead'] * layers

    logger.info("    Localization mem time: " +
        str(dataLoadAndStoreTime * 1.0e6) + " us, math time: " + str(totalMathTime * 1.0e6) +
        " us, total time: " + str(totalTime * 1.0e6) + " us (" + str(totalFlops / (totalTime * 1.0e12)) +
        " TFLOP/s) (" + str(gpuPerformance['kernel-overhead'] * 100. / kernelTime) + "% sync overhead)")

    return totalTime

def estimateCoordinateTransformPerformanceWithRooflineModel(imageSize, gpuPerformance):
    coordinateBytes  = (getInputSize(imageSize) * 4) * gpuPerformance['word-size'] * gpuPerformance['mini-batch-size']
    thetaBytes = (4*3) * gpuPerformance['word-size'] * gpuPerformance['mini-batch-size']

    totalBytes = (coordinateBytes + thetaBytes)

    dataLoadAndStoreTime = totalBytes / (gpuPerformance['bandwidth'] * gpuPerformance['cublas-memory-efficiency'])

    totalFlops = gpuPerformance['mini-batch-size'] * getTotalImageSize(imageSize) * 3 * 4 * 2

    totalThreads = gpuPerformance['sms'] * gpuPerformance['datapaths-per-sm']

    flopsPerThread = totalFlops / totalThreads

    utilization = min(gpuPerformance['work-per-datapath'], flopsPerThread) / gpuPerformance['work-per-datapath']

    totalMathTime = totalFlops / (gpuPerformance['flops'] * utilization * gpuPerformance['cublas-math-efficiency'])

    kernelTime = max(totalMathTime, dataLoadAndStoreTime)

    totalTime = kernelTime + gpuPerformance['kernel-overhead']

    logger.info("    Coordinate transform mem time: " +
        str(dataLoadAndStoreTime * 1.0e6) + " us, math time: " + str(totalMathTime * 1.0e6) +
        " us, total time: " + str(totalTime * 1.0e6) + " us (" + str(totalFlops / (totalTime * 1.0e12)) +
        " TFLOP/s) (" + str(gpuPerformance['kernel-overhead'] * 100. / kernelTime) + "% sync overhead)")

    return totalTime

def estimateBilinearFilteringPerformanceWithRooflineModel(imageSize, gpuPerformance):
    totalCopies = (3. + # clamp to input boundary
                  8. + # get sample positions
                  8*2./4 + # get batch ids
                  12 +  # getNeighborhood weights
                  16 + # gathered data
                  16 # multiplication with weights
                  )

    imageBytes = (getInputSize(imageSize)) * gpuPerformance['word-size'] * gpuPerformance['mini-batch-size']

    totalBytes = imageBytes * totalCopies

    dataLoadAndStoreTime = totalBytes / (gpuPerformance['bandwidth'] * gpuPerformance['cublas-memory-efficiency'])

    totalFlops = getInputSize(imageSize) * totalCopies

    totalThreads = gpuPerformance['sms'] * gpuPerformance['datapaths-per-sm']

    flopsPerThread = totalFlops / totalThreads

    utilization = min(gpuPerformance['work-per-datapath'], flopsPerThread) / gpuPerformance['work-per-datapath']

    totalMathTime = totalFlops / (gpuPerformance['flops'] * utilization * gpuPerformance['memory-bound-math-efficiency'])

    kernelTime = max(totalMathTime, dataLoadAndStoreTime)

    totalTime = kernelTime + gpuPerformance['kernel-overhead'] * totalCopies

    logger.info("    Bilinear filter mem time: " +
        str(dataLoadAndStoreTime * 1.0e6) + " us, math time: " + str(totalMathTime * 1.0e6) +
        " us, total time: " + str(totalTime * 1.0e6) + " us (" + str(totalFlops / (totalTime * 1.0e12)) +
        " TFLOP/s) (" + str(gpuPerformance['kernel-overhead'] * 100. / kernelTime) + "% sync overhead)")

    return totalTime

def estimatePerformanceWithRooflineModel(imageSize, filterSize, layers, gpuPerformance):
    totalTime = estimateLocalizationNetworkPerformanceWithRooflineModel(imageSize, filterSize, layers, gpuPerformance)
    totalTime += estimateCoordinateTransformPerformanceWithRooflineModel(imageSize, gpuPerformance)
    totalTime += estimateBilinearFilteringPerformanceWithRooflineModel(imageSize, gpuPerformance)
    logger.info("    Total time " + str(totalTime))

def generateEstimates():
    gpuPerformance = {
        "flops"     : 6.144e12,
        "bandwidth" : 336.e9,
        "sms" : 24,
        "datapaths-per-sm" : 128,
        "work-per-datapath" : 1000.,
        "kernel-overhead" : 10.0e-6,
        "word-size" : 4,
        "cudnn-math-efficiency" : 0.5,
        "cudnn-memory-efficiency" : 0.25,
        "cublas-math-efficiency" : 0.8,
        "cublas-memory-efficiency" : 0.33,
        "memory-bound-memory-efficiency" : 0.8,
        "memory-bound-math-efficiency" : 0.05,
        "mini-batch-size" : 64
    }

    targetMiniBatchSize = 64.

    filterSizes = [1, 3, 5]
    localizationLayers = [1, 2, 3]
    imageSizes = [[256, 256, 3, 64], [128, 128, 64, 64], [64, 64, 64, 128], [32, 32, 128, 128], [16, 16, 128, 256], [8, 8, 256, 256], [4, 4, 256, 256]]

    for filterSize in filterSizes:
        logger.info("For filter size " + str(filterSize))
        for layerCount in localizationLayers:
            logger.info(" For layer count " + str(layerCount))
            for imageSize in imageSizes:
                logger.info("  For image size " + str(imageSize))
                estimatePerformanceWithRooflineModel(imageSize, filterSize, layerCount, gpuPerformance)





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


if __name__ == '__main__':
    main()
