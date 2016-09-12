###############################################################################
#
# \file    subMatrixAllOnes.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Sunday August 14, 2016
# \brief   Given a matrix of dimensions mxn having all entries as 1 or 0, 
#          find out the size of maximum size square sub-matrix with all 1s
#
###############################################################################

import os
import argparse
import logging
import json
import time
import numpy

# bottom up
def findSubMatrixAllOnes(inputM, cache):
    maxSize = 0
    rows, cols = inputM.shape
    #bottoms up approach - hence iterations, usually polynomial
    for i in range (0, rows):
        for j in range (0, cols):
            #base cases memoized
            if i == 0 or j == 0:
                cache[str(i) + ',' + str(j)] = inputM[i][j]
            else:
                if inputM[i][j] == 0:
                    cache[str(i) + ',' + str(j)] = 0
                #used cache instead of recursive calls
                if inputM[i][j] == 1:
                    cache[str(i) + ',' + str(j)] = min(cache.get(str(i-1)+','+str(j)), cache.get(str(i)+','+str(j-1)), cache.get(str(i-1)+','+str(j-1))) + 1
                #current answer
                if cache.get(str(i)+','+str(j)) > maxSize:
                    maxSize = cache.get(str(i)+','+str(j))
    return maxSize


def main():
    parser = argparse.ArgumentParser(description="Biggest sub matrix")
    parser.add_argument("-v", "--verbose",        default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    s = 4
    inputData = numpy.zeros((s, s))
    inputData[0, 1] = 1
    inputData[0, 2] = 1
    inputData[0, 3] = 1
    inputData[1, 1] = 1
    inputData[1, 2] = 1
    inputData[1, 3] = 1
    inputData[2, 1] = 1
    inputData[2, 2] = 1
    inputData[2, 3] = 1



    cache = {}
    print findSubMatrixAllOnes(inputData, cache)

if __name__ == '__main__':
    main()
