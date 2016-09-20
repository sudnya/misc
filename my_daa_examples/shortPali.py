# Enter your code here. Read input from STDIN. Print output to STDOUT
import sys

def findMatchingChar(c, string1, newStart, newEnd):
    for i in range(newStart, newEnd):
        if c == string1[i]:
            #print "found match for ", c, " at offset ", i
            return i
    return -1

def getTotalPali(string1):
    retVal = set()
    for j in range(0, len(string1)):
        for i in range(0, len(string1)):
            hasFirst = findMatchingChar(string1[i], string1, i+2, len(string1))
            a = i
            d = hasFirst + a 
            
            print "a", a, " d ", d
            
            if d > i:
                hasSecond = findMatchingChar(string1[i+1], string1, i+2, d)
                b = i + 1
                c = hasSecond + a 
                
                print "b", b, " c ", c
                
            if a < b and b < c and c < d:
                retVal.add((a, b, c, d))
                print "added ", retVal
    temp = len(retVal)%(pow(10, 9)+7)
    print (temp)

def getShortPali(str1):
    a = i
    

def main():
    testStr = raw_input()
    #testStr = "kkkkkkz"
    #print testStr
    getTotalPali(testStr)
    
if __name__ == '__main__':
    main()
