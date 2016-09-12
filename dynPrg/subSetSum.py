###############################################################################
#
# \file    subSetSum.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Sunday August 14, 2016
# \brief   Detect if a subset from a given set of N non-negative integers sums 
#          upto a given value S  
#
###############################################################################

import os
import argparse
import logging
import json
import time
import numpy

logger = logging.getLogger('subsetsum')

# inputdata, size of input to consider, testCondition, cache
def isSubsetSumPossible(inputD, maxPos, expectedS, cache):

    #unique key for each possible subproblem
    key = (maxPos, expectedS)

    if key in cache:
        return cache[key]

    if maxPos < 0:
        return False

    #not possible to include this input value, just update position
    if inputD[maxPos] > expectedS:
        result = isSubsetSumPossible(inputD, maxPos-1, expectedS, cache)
        #logger.debug("Result (" + str((maxPos-1, expectedS)) + ") : " + str(result))
        return result

    # termination - found at least a set of one
    if inputD[maxPos] == expectedS:
        #logger.debug("result (" + str((maxPos, expectedS)) + ") : " + str(True))
        return True

    #we could either use this current value or not, either way it is considered so use OR
    includedResult    = isSubsetSumPossible(inputD, maxPos-1, expectedS - inputD[maxPos], cache)
    notIncludedResult = isSubsetSumPossible(inputD, maxPos-1, expectedS, cache)

    result = includedResult or notIncludedResult

    logger.debug("Caching result (" + str((maxPos, expectedS)) + ") : " + str(result))

    cache[(maxPos, expectedS)] = result

    return result

# bottom up approach, iterative
def iterativeSubsetSum(inputD, expectedS, cache):
    for i in range(0, len(inputD)):
        for j in range(0, expectedS+1): # +1 since we want to include expectedS
            logger.debug("iter: " + str(i) + " , " + str(j))
            if i == 0 and j == 0:
                cache[(i,j)] = True #both empty sets and 0
            elif j == 0:
                cache[(i,j)] = True #0 can be made by getting empty set
            elif i == 0:
                cache[(i,j)] = False #can never get any expectedS with just empty set
            else:
                if inputD[i] == j: 
                    cache[(i,j)] = True #if equal exactly, they this number alone can make one element set of expectedS
                else:
                    cache[(i,j)] = cache[(i-1, j)] or cache[(i-1, j-i)] #key! look for other number at j-i to get leftover expectedS
    return cache[len(inputD)-1, expectedS]


def main():
    parser = argparse.ArgumentParser(description="Subset sum exists?")
    parser.add_argument("-v", "--verbose",        default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    expectedS = 5
    inputData = [1,3,9,2]
    cache = {}
    print isSubsetSumPossible(inputData, len(inputData)-1, expectedS, cache)
    cache2 = {}
    print iterativeSubsetSum(inputData, expectedS, cache)


if __name__ == '__main__':
    main()
