###############################################################################
#
# \file    bst.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Saturday Aug 27, 2016
# \brief   binary search tree
#
###############################################################################


def searchBst(inputA, query):
    lo = 0
    hi = len(inputA) - 1

    while lo <= hi:
        mid = lo + (hi - lo)/2
        print "Searching at: ", mid
        if inputA[mid] == query:
            return True
        if query > inputA[mid]:
            lo = mid+1
            print query , " > ", inputA[mid], " so go right new lo: ", lo
        if query < inputA[mid]:
            hi = mid-1
            print query , " < ", inputA[mid], " so go left new hi: ", hi
    return False
    
#first element greater than or equal to query
def searchLowerBound(inputA, query):
    lo = 0
    hi = len(inputA)

    diff = hi - lo

    while diff > 0:
        diff = hi - lo
        mid = lo + (diff/2)
        print "diff: ", diff, " mid: ", mid, " high: ", hi , " , low: ", lo

        if query > inputA[mid]:
            print "Searching at offset: ", mid , " contains ", query , " > " , inputA[mid] , " go right"
            lo = mid + 1
        else:
            print "Searching at offset: " , mid , " contains ", query , " <= " , inputA[mid] , " go left"
            hi = mid
    return lo


#first element greater than query
def searchUpperBound(inputA, query):
    lo = 0
    hi = len(inputA)
    diff = hi - lo

    while diff > 0:
        diff = hi - lo
        mid  = lo + (diff/2)
        print "diff: ", diff, " mid: ", mid, " high: ", hi , " , low: ", lo

        # if query is higher, look on the right
        if not (query < inputA[mid]):
            print "Searching at offset: ", mid , " contains ", query , " !< " , inputA[mid] , " go right"
            lo = mid + 1

        else:
            #left
            print "Searching at offset: " , mid , " contains ", query , " <= " , inputA[mid] , " go left"
            hi = mid

    print "returning ", lo
    return lo



def main():
    #inputA = [1,2,4,8,16,32,64]
    #query = 4
    #found = searchBst(inputA, query)
    #print "Found = " , found , " element = ", query, " in list: ", inputA
    #query = 6 
    #found = searchBst(inputA, query)
    #print "Found = " , found , " element = ", query, " in list: ", inputA

    inputB = [1,2,5,5,5,7,8,9]
    lb = 5
    x  = searchLowerBound(inputB, lb)
    print "lower bound at ", x, " for query of ", lb , " in list: ", inputB

    x  = searchUpperBound(inputB, lb)
    print "upper bound at ", x, " for query of ", lb , " in list: ", inputB

if __name__ == '__main__':
    main()
