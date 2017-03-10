###############################################################################
#
# \file    moveZerosToEnd.py
# \author  Sudnya Diamos <mailsudnya@gmail.com>
# \date    Friday March 10, 2017
# \brief   Given an array of numbers, convert it into an array with the 0s
#          moved to the end of the array. Preserve ordering for the rest of
#          the elements
###############################################################################

import argparse
import logging

logger = logging.getLogger('moveZerosToEnd')

def shiftLeft(inputs, idx):
    for i in range(idx, len(inputs)-1):
        inputs[i] = inputs[i+1]
    return inputs
    

def moveZerosToEnd(inputs):
    # find total number of zeros
    numZeros = 0
    for i in range(len(inputs)):
        if inputs[i] == 0:
            numZeros += 1
    logger.debug (inputs)
    
    # shift all non zero inputs in order to left
    ctr = 0
    for i in range(len(inputs)):
        while inputs[i] == 0:
            logger.debug ("shift idx " + str(i))
            shiftLeft(inputs, i)
            ctr += 1
            logger.debug(inputs)
            if ctr > len(inputs):
                break

    #put zeros at last numZeros location
    for j in range(len(inputs)-numZeros, len(inputs)):
        inputs[j] = 0
    
    return inputs

def compareRef(ref, o):
    #ensure len is same
    logger.debug("ref " + str(len(ref)) + " while output is of len " + str(len(o)))
    assert len(ref) == len(o)

    #ensure contents are the same
    for i in range(len(ref)):
        assert ref[i] == o[i]

def runTest():
    t1 = [0,1,4,0,9,0,0,0,1,2]
    r1 = [1,4,9,1,2,0,0,0,0,0]
    o1 = moveZerosToEnd(t1)
    compareRef(r1, o1)
    logger.info ("Test with input " + str(t1) + " passed reference check. Output " + str(o1))
   
    #only one entry - 0
    t2 = [0]
    r2 = [0]
    o2 = moveZerosToEnd(t2)
    compareRef(r2, o2)
    logger.info ("Test with input " + str(t2) + " passed reference check. Output " + str(o2))

    #only  0s
    t3 = [0, 0, 0]
    r3 = [0, 0, 0]
    o3 = moveZerosToEnd(t3)
    compareRef(r3, o3)
    logger.info ("Test with input " + str(t3) + " passed reference check. Output " + str(o3))

    #only one entry
    t4 = [4]
    r4 = [4]
    o4 = moveZerosToEnd(t4)
    compareRef(r4, o4)
    logger.info ("Test with input " + str(t4) + " passed reference check. Output " + str(o4))

    #only non zeros 
    t5 = [1, 2, 3]
    r5 = [1, 2, 3]
    o5 = moveZerosToEnd(t5)
    compareRef(r5, o5)
    logger.info ("Test with input " + str(t5) + " passed reference check. Output " + str(o5))

    # already has zeros at the end
    t6 = [1,4,9,1,2,0,0,0,0,0]
    r6 = [1,4,9,1,2,0,0,0,0,0]
    o6 = moveZerosToEnd(t6)
    compareRef(r6, o6)
    logger.info ("Test with input " + str(t6) + " passed reference check. Output " + str(o6))
    
    t7 = []
    r7 = []
    o7 = moveZerosToEnd(t7)
    compareRef(r7, o7)
    logger.info ("Test with input " + str(t7) + " passed reference check. Output " + str(o7))
    
    t8 =  [0,0,0,1,1,1]
    r8 = [1,1,1,0,0,0]
    o8 = moveZerosToEnd(t8)
    compareRef(r8, o8)
    logger.info ("Test with input " + str(t8) + " passed reference check. Output " + str(o8))
   
    logger.info("Passed all reference checks")



def main():
    parser = argparse.ArgumentParser(description="Move Zeros To End")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    runTest()


   

if __name__ == '__main__':
    main()
