
import sys
import math

def isValid(successor, blocks, maxR, maxC):
    row, column = successor

    if row < 0 or column < 0 or row >= maxR or column >= maxC:
        return False

    if successor in blocks:
        return False

    return True

def getSuccessors(position, blocks, maxR, maxC):
    row, column = position

    left  = (row, column-1)
    right = (row, column+1)
    up    = (row-1, column)
    down  = (row+1, column)

    possibleSuccessors = [left, right, up, down]

    return [successor for successor in possibleSuccessors if isValid(successor, blocks, maxR, maxC)]

def getGuardDist(blocks, g, maxR, maxC):
    visited = set(g)

    frontier = []
    frontier.extend(g)
    
    distance = 0
    while len(frontier) != 0:
        newFrontier = []
        for i in frontier:
            print i, distance

            successors = getSuccessors(i, blocks, maxR, maxC)

            for successor in successors:
                if not successor in visited:
                    visited.add(successor)
                    newFrontier.append(successor)

        distance += 1
        frontier = newFrontier



def main():
    maxR = 4
    maxC = 4

    blockList = []
    blockList.append((1,0))
    blockList.append((1,1))

    guardList = []
    guardList.append((2,1))
    guardList.append((2,3))
    guardList.append((0,2))

    getGuardDist(set(blockList), guardList, maxR, maxC)

if __name__ == '__main__':
    main()

