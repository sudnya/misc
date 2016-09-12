###############################################################################
#
# \file    heapStuff.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Saturday Aug 27, 2016
# \brief   heap sort using min and max heap
#
###############################################################################
import argparse
import math
import sys
import logging

logger = logging.getLogger("heapStuff")

#class heap: insert(), getMax(), extractMax(), updateKey()
class maxHeap():
    def __init__(self):
        self.elements = [sys.maxint]

    def insert(self, entry):
        self.elements.append(entry)
        logger.debug ("just inserted entry to heap: " + str(entry))
        self.heapifyUp()


    def getMax(self):
        return self.elements[1]

    def extractMax(self):
        # get the max
        retVal = self.elements[1]
        # swap root with last leaf
        self.swap(1, len(self.elements)-1)
        # delete the new last leaf
        logger.debug ("removing " + str(retVal))
        del self.elements[-1]

        #rebalance
        self.heapifyDown()
        return retVal


    def updateKey(self):
        pass

    def swap(self, parent, child):
        temp                  = self.elements[parent]
        self.elements[parent] = self.elements[child]
        self.elements[child]  = temp


    def heapifyDown(self):
        parent     = 1
        leftChild  = 2*parent
        rightChild = (2*parent) + 1
        maxIdx     = parent

        while True:
            if parent > (len(self.elements)-1):
                break

            if leftChild <= len(self.elements) - 1 and self.elements[leftChild] > self.elements[maxIdx]:
                logger.debug("left child [" + str(leftChild) + "] : " + str(self.elements[leftChild]) + " bigger")
                maxIdx = leftChild
            if rightChild <= len(self.elements) - 1 and self.elements[rightChild] > self.elements[maxIdx]:
                logger.debug("right child [" + str(rightChild) + "] : " + str(self.elements[rightChild]) + " bigger")
                maxIdx = rightChild

            if maxIdx != parent:
                logger.debug("defied max heap, swapping [" + str(parent) + "]: " + str(self.elements[parent]) + " with [" + str(maxIdx) + "]: " + str(self.elements[maxIdx]))
                self.swap(parent, maxIdx)
            parent     = parent * 2
            leftChild  = 2*parent
            rightChild = (2*parent) + 1
            maxIdx     = parent
    
    def heapifyUp(self):
        parent     = (len(self.elements) - 1)/2
        leftChild  = 2*parent
        rightChild = (2*parent) + 1
        maxIdx     = parent

        while True:
            if parent < 1:
                break

            if leftChild <= len(self.elements) - 1 and self.elements[leftChild] > self.elements[maxIdx]:
                logger.debug("left child [" + str(leftChild) + "] : " + str(self.elements[leftChild]) + " bigger")
                maxIdx = leftChild
            if rightChild <= len(self.elements) - 1 and self.elements[rightChild] > self.elements[maxIdx]:
                logger.debug("right child [" + str(rightChild) + "] : " + str(self.elements[rightChild]) + " bigger")
                maxIdx = rightChild

            if maxIdx != parent:
                logger.debug("defied max heap, swapping [" + str(parent) + "]: " + str(self.elements[parent]) + " with [" + str(maxIdx) + "]: " + str(self.elements[maxIdx]))
                self.swap(parent, maxIdx)
            parent     = parent / 2
            leftChild  = 2*parent
            rightChild = (2*parent) + 1
            maxIdx     = parent


        logger.debug ("new status of heap: ")
        for i in range(0, len(self.elements)):
            logger.debug (self.elements[i])


class minHeap():
    def __init__(self):
        self.elements = [sys.maxint]

    def insert(self, entry):
        self.elements.append(entry)
        logger.debug ("just inserted entry to heap: " + str(entry))
        self.heapifyMinUp()

    def getMin(self):
        return self.elements[1]

    def extractMin(self):
        retVal = self.elements[1]
        self.swap(1, len(self.elements)-1)
        del self.elements[-1]
        self.heapifyMinDown()
        return retVal

    def swap(self, i, j):
        temp = self.elements[i]
        self.elements[i] = self.elements[j]
        self.elements[j] = temp


    def heapifyMinDown(self):
        parent     = 1
        leftChild  = parent * 2
        rightChild = (parent * 2) + 1
        minIdx     = parent

        while True:
            if parent > (len(self.elements)-1):
                break

            if leftChild <= len(self.elements) - 1 and self.elements[leftChild] < self.elements[minIdx]:
                minIdx = leftChild
            if rightChild <= len(self.elements) - 1 and self.elements[rightChild] < self.elements[minIdx]:
                minIdx = rightChild

            if minIdx != parent:
                self.swap(parent, minIdx)
            parent     = parent * 2
            leftChild  = parent * 2
            rightChild = (parent*2) + 1
            minIdx     = parent
        logger.debug ("new status of heap: ")
        for i in range(0, len(self.elements)):
            logger.debug (self.elements[i])

    def heapifyMinUp(self):
        parent     = (len(self.elements) - 1 ) / 2 
        leftChild  = parent * 2
        rightChild = (parent * 2) + 1
        minIdx     = parent

        while True:
            if minIdx < 1:
                break

            if leftChild <= len(self.elements) - 1 and self.elements[leftChild] < self.elements[minIdx]:
                logger.debug("left child [" + str(leftChild) + "] : " + str(self.elements[leftChild]) + " smaller")
                minIdx = leftChild
            if rightChild <= len(self.elements) - 1 and self.elements[rightChild] < self.elements[minIdx]:
                logger.debug("right child [" + str(rightChild) + "] : " + str(self.elements[rightChild]) + " smaller")
                minIdx = rightChild

            if minIdx != parent:
                self.swap(minIdx, parent)
            parent     = parent / 2
            leftChild  = parent * 2
            rightChild = (parent*2) + 1
            minIdx     = parent
        logger.debug ("new status of heap: ")
        for i in range(0, len(self.elements)):
            logger.debug (self.elements[i])

# sort descending order
def runMaxHeapSort():
    #input array
    a = [9, 1, 3, 4, 11, 2, 5, 13, 17, 7]
    # build a max heap
    customHeap = maxHeap()
    for i in a:
        customHeap.insert(i)

    #run extract till heap is empty - descending order
    print "\n\n\n\n\nsorted descending"
    for i in range(0, len(a)):
        x = customHeap.extractMax()
        print x

# ascending order
def runMinHeapSort():
    #input array
    a = [9, 1, 3, 4, 11, 2, 5, 13, 17, 7]
    # build a min heap
    customHeap = minHeap()
    for i in a:
        customHeap.insert(i)

    #run extract till heap is empty - ascending order
    print "\n\n\n\n\nsorted Ascending:"
    for i in range(0, len(a)):
        x = customHeap.extractMin()
        print x


def main():
    parser = argparse.ArgumentParser(description="Implementing min/max heap ")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    runMaxHeapSort()
    runMinHeapSort()


if __name__ == '__main__':
    main()
