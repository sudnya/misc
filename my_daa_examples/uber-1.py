## This is the text editor interface. 
## Anything you type or change here will be seen by the other person in real time.

# Integer: number of courses  (3) (0, 1, 2)
# List  of tuples: ((0, 1), (1, 2))  # 2 prerequisites: (1 must be taken before 0), (2 muts be taken before 1)

# 2-->1-->0

# Boolean: whether it is possible to take all courses
from collections import deque

def explore(node, graph, visited, myStack):
    visited.add(node) #-> (0,1,2)
    deps = graph.get(node) #none
    
    if deps == None or len(deps) ==0:
        myStack.append(node) #stack -> 2
        return
    
    for child in deps:
        #unexplored child, explore it's children
        if child not in visited:
            explore(child, graph, visited, myStack)
    
    # done exploring all children
    myStack.append(node) #2, 1

def doTopoSort(graph, startNode):
    visited = set()
    myStack = deque()
    
    visited.add(startNode) #-> 0
    children = graph.get(startNode) #-> 1
    
    currentNode = startNode
    
    
    while len(visited) < len(graph.keys()):
        children = graph.get(currentNode)
        currentNode -= 1
        if currentNode < 0:
            break
        
        if children == None:
            myStack.appendleft(currentNode)
            continue
        
        for child in children:
            if child == None:
                break
                #continue #todo: break? if no child
            if child not in visited:
                explore(child, graph, visited, myStack)
    
    myStack.appendleft(startNode)
    print myStack
    return myStack

def main():
    totalCourses = 3
    deps = []
    deps.append((0,1))
    deps.append((1,2))
    
    # 4--->1
    # 0--->1--->2--->3
    
    #  Assert.assertTrue(canFinish(5, new int[][] {{1, 4}, {1,0}, {2,1}, {3, 2}}));
    adjList = {}
    # (0, 1)
    # (2, 1)
    # 2, 1, 0 -> true
    #adjList[0] = set([1]) # require 1 before 0
    #adjList[1] = set([2]) # it is required for 0 to be taken before 1
    #adjList[2] = set([3])
    #adjList[3] = None
    #adjList[4] = set([1])
    totalCourses = 3
    n = totalCourses - 1
    adjList[0] = set([1])
    adjList[1] = set([2])
    adjList[2] = set([0])
    # 0-->1--->2
    #  |\_____/
    # Assert.assertFalse(canFinish(3, new int[][] {{1,0}, {2, 1}, {0, 2}}));
    temp = {}
    temp[0] = set([2])
    temp[1] = set([0])
    temp[2] = set([1])
    sortedNodes = doTopoSort(temp, n)
    if len(sortedNodes) < len(temp.keys()):
        print False
    else:
        print True
        


if __name__ == '__main__':
    main()
