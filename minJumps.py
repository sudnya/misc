#Given an array of integers where each element represents the max number of steps that can be made forward from that element. Write a function to return the minimum number of jumps to reach the end of the array (starting from the first element). If an element is 0, then cannot move through that element.
#
#Example:
#
#    Input: arr[] = {1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9}
#    Output: 3 (1-> 3 -> 8 ->9)
#    First element is 1, so can only go to 3. Second element is 3, so can make at most 3 steps eg to 5 or 8 or 9.
import sys

def getMinJumps(inputA, cache):
    if len(inputA) == 0:
        return 0
    if inputA[0] == 0:
        return 0

    for k in range(0, len(inputA)):
        cache[k] = sys.maxint

    cache[0] = 0
    for i in range(1, len(inputA)):
        #only look what is reacheable upto i
        for j in range(0, i):
            # if current offset + contents at offset is more that when we want to reach
            if j + inputA[j] >= i:
                cache[i] = min(cache[i], cache[j] + 1)
                print j , "," , j , ": " , inputA[j] , " >= ", i, " so possible, cache[", i, "]" , cache[i]
    return cache[len(inputA)-1]
            

def main():
    #inputArray = [1,4,3,7,1,2]
    inputArray = [1,3,5,8,9,2,6,7,6,8,9]
    cache = {}
    minJumps = getMinJumps(inputArray, cache)
    print "Min jumps" , minJumps 

if __name__ == '__main__':
    main()
