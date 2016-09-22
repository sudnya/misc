###############################################################################
#
# \file    subMatrixAllOnes.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Monday August 15, 2016
# \brief   Given two strings S1 and S2, find out the minimum edit distance to 
#          transform S1 to S2. 
#          Edit operations: insertion, deletion, substitution
#
###############################################################################

import os
import argparse
import logging
import json
import time
import numpy

logger = logging.getLogger('minEditDistance')

def findMinEditDistance(s1, s1pos, s2, s2pos, cache):
    if s1pos < 0 and s2pos < 0:
        return 0

    #if s1pos is -1 then to correct it to 0, s2pos would be +1
    if s1pos < 0:
        return s2pos + 1
    if s2pos < 0:
        return s1pos + 1
    key = str(s1pos) + "_" + str(s2pos)

    if key in cache:
        logger.debug("found " + key + " in cache : " + str(cache[key]))
        return cache[key]

    
    if s1[s1pos] == s2[s2pos]:
        logger.debug("equal chars at pos: " + str(s1pos) + " and " + str(s2pos))
        cache[key] = findMinEditDistance(s1, s1pos-1, s2, s2pos-1, cache)
    else:
        cache[key] = 1 + min(findMinEditDistance(s1, s1pos-1, s2, s2pos, cache), findMinEditDistance(s1, s1pos, s2, s2pos-1, cache),  findMinEditDistance(s1, s1pos-1, s2, s2pos-1, cache))
    
    logger.debug("key:" + key +" Comparing " + s1[s1pos] + " to " + s2[s2pos] + " : " + str(cache[key]))

    return cache[key]

def printCache(cache, maxR, maxC):
    logger.info("Memoized table: ")
    for i in range(0, maxR):
        for j in range(0, maxC):
            print cache[(i, j)] ,
        print "\n"


# table: s1 rows and s2 cols
def iterativeMinEditDistance(s1, s2, cache):
    for i in range(0, len(s1)):
        for j in range(0, len(s2)):
            # base case
            if i == 0 or j == 0:
                # equal
                if s1[i] == s2[j]:
                    cache[(i, j)] = 0
                # requires substitution/insert/delete
                else:
                    cache[(i, j)] = 1 #+ max(i, j) # one char in s1 - takes 
                    logger.debug(str(i) + ", " + str(j) + " cache : " + str(cache))
            else:
                # equal, so no need to add any edit cost
                if s1[i] == s2[j]:
                    cache[(i, j)] = min(cache[(i-1, j-1)], cache[(i-1, j)], cache[(i, j-1)])
                # one more op in addition to the edit cost
                else:
                    cache[(i, j)] = 1 + min(cache[(i-1, j-1)], cache[(i-1, j)], cache[(i, j-1)])
            
    printCache(cache, len(s1), len(s2))
    return cache[len(s1)-1, len(s2)-1]


def main():
    parser = argparse.ArgumentParser(description="Minimum Edit Distance")
    parser.add_argument("-v", "--verbose",        default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    inputS1 = "kitten"
    inputS2 = "sitting"

    pos1 = len(inputS1) - 1
    pos2 = len(inputS2) - 1
    cache = {}
    editD = findMinEditDistance(inputS1, pos1, inputS2, pos2, cache)
    logger.info("Edit distance of " + inputS1 + " and " + inputS2 + " is = " + str(editD))
    
    cache1 = {}
    editD1 = iterativeMinEditDistance(inputS1, inputS2, cache1)
    logger.info("Edit distance of " + inputS1 + " and " + inputS2 + " is = " + str(editD1))

if __name__ == '__main__':
    main()
