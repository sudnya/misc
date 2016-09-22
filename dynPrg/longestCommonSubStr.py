###############################################################################
#
# \file    longestCommonSubStr.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Monday August 15, 2016
# \brief   Given two strings S1 and S2, find out the longest common substring
#
###############################################################################
#TODO: redo this in a simple way

import os
import argparse
import logging
import json
import time
import numpy

logger = logging.getLogger('longestCommonSubStr')

# bottom up
def findLongestCommonSubString(s1, s2, cache):

    # get a truth table (with length!) for whether or not this char makes it into a substr
    for i in range(0, len(s1)):
        for j in range(0, len(s2)):
            if i == 0 or j == 0:
                if s1[i] == s2[j]:
                    cache[(i, j)] = 1
                else:
                    cache[(i, j)] = 0
            else:
                if s1[i] == s2[j]:
                    cache[(i, j)] = 1 + cache[(i-1, j-1)]
                else:
                    cache[(i, j)] = 0
    
    #find the biggest length location
    maxV = 0
    maxLoc = []
    for i in range(0, len(s1)):
        for j in range(0, len(s2)):
            val = cache[(i, j)]
            if val > maxV:
                maxV = val
                maxLoc = []

            if val == maxV:
                maxLoc.append((i,j))

    for i, j in maxLoc:
        string = ""
        while True:
            if i < 0 or j < 0:
                break

            value = cache[(i,j)]

            if value == 0:
                break

            string += s1[i]

            i -= 1
            j -= 1

        print string


def main():
    parser = argparse.ArgumentParser(description="Longest common subsequence")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    inputS1 = "LCLC"
    inputS2 = "CLCL"
    #longest common substr CLC, LCL

    pos1 = len(inputS1) - 1
    pos2 = len(inputS2) - 1
    cache = {}
    lSubString = findLongestCommonSubString(inputS1, inputS2, cache)
    logger.info("Longest common substring: " + lSubString)

if __name__ == '__main__':
    main()
