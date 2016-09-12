###############################################################################
#
# \file    longestCommonSubSeq.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Monday August 15, 2016
# \brief   Given two strings S1 and S2, find out the longest common substring
#
###############################################################################

import os
import argparse
import logging
import json
import time
import numpy

logger = logging.getLogger('longestCommonSubSeq')

def findLongestCommonSubSequence(s1, s1pos, s2, s2pos, cache):
    #termination condition
    if s1pos >= len(s1) and s2pos >= len(s2):
        return 0
    if s1pos >= len(s1):
        return 0
    if s2pos >= len(s2):
        return 0

    #unique key
    key = (s1pos, s2pos)
    #is it in cache?
    if key in cache:
        logger.debug("added key: " + str(key) + " to cache")
        return cache[key]


    #if match found, add to list and kick off partial subproblem
    if s1[s1pos] == s2[s2pos]:
        logger.debug("pos1: " + str(s1pos) + " in " + s1 + " matches pos2: " + str(s2pos) + " in " + s2)
        cache[key] = 1 + findLongestCommonSubSequence(s1, s1pos+1, s2, s2pos+1, cache)

    #else, kick off multiple subproblems with various options
    else:
        cache[key] = max(findLongestCommonSubSequence(s1, s1pos, s2, s2pos+1, cache), findLongestCommonSubSequence(s1, s1pos+1, s2, s2pos, cache))

    logger.info("cache contains: " + str(cache))
    return cache[key]

    


def main():
    parser = argparse.ArgumentParser(description="Longest common subsequence")
    parser.add_argument("-v", "--verbose",        default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    inputS1 = "kit"#ten"
    inputS2 = "sit"#ting"
    #longest common subseq ittn

    cache = {}
    lSubSeq = findLongestCommonSubSequence(inputS1, 0, inputS2, 0, cache)
    logger.info("Longest common subequence : " + str(lSubSeq))

if __name__ == '__main__':
    main()
