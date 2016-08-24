#If you are given an integer array and an integer 'k' as input, write a program to print elements with maximum values from each possible sub-array (of given input array) of size 'k'. 
#If the given input array is {9,6,11,8,10,5,14,13,93,14} and for k = 4, 
# output should be 11,11,11,14,14,93,93. 
#Please observe that 11 is the largest element in the first, second and third sub-arrays - {9,6,11,8}, {6,11,8,10} and {11,8,10,5}; 
#14 is the largest element for fourth and fifth sub-array and 93 is the largest element for remaining sub-arrays.
#

import sys
import collections

def printMax(array, k):
    d = collections.deque()
    
    #init deque
    for i in range(0, k):
        for j in range(0, len(d)):
            if array[i] > array[j]:
                d.popleft()
        d.append(i)

    #print "\n" , d, "\n"
    for i in range(k, len(array)):
        # print max
        print array[d[0]]
        
        #resize the current window
        for j in range(0, len(d)):
            if d[j] < i - k:
                d.popleft()

        while len(d) != 0 and array[i] > array[d[-1]]:
            d.pop()

        # add this element
        d.append(i)
    print array[d[0]]

def main():
    array = [9,6,11,8,10,5,14,13,93,14]
    k = 4
    printMax(array, k)
    #maxList = []
    #for i in range(0, len(array)-k+1):
    #    maxInIter = 0
    #    for j in range(i, i+k):
    #        if array[j] > maxInIter:
    #            maxInIter = array[j]
    #    maxList.append(maxInIter)

    #for k in range(len(maxList)):
    #    print maxList[k] , "\t"


if __name__ == '__main__':
    main()
