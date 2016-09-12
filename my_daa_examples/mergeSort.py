###############################################################################
#
# \file    mergeSort.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Friday Aug 26, 2016
# \brief   merge sort
#
###############################################################################

def merge(left, right):
    if not left:
        return right
    if not right:
        return left
    if left[0] < right[0]:
        return [left[0]] + merge(left[1:], right)
    return [right[0]] + merge(left, right[1:])
    

def runMergeSort(inputA):
    if len(inputA) <= 1:
        return inputA

    mid   = len(inputA) // 2
    left  = runMergeSort(inputA[:mid])
    right = runMergeSort(inputA[mid:])
    return merge(left, right)

def main():
    inputA       = [5,1,4,99,11,13]
    sortedOutput = runMergeSort(inputA)
    
    for i in range(0, len(sortedOutput)):
        print sortedOutput[i]

if __name__ == '__main__':
    main()
