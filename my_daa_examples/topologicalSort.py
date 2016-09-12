###############################################################################
#
# \file    topologicalSort.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Saturday Aug 27, 2016
# \brief   topological sort
#
###############################################################################

import argparse
import logging
from collections import deque

logger = logging.getLogger("topologicalSort")

def runTopologicalSort(graph, startingNode):
    visited = set()
    myStack = deque()

    visited.add(startingNode)

    logger.debug ("Consider: " + startingNode)
    
    children = graph.get(startingNode)
    logger.debug ("children: " + str(children))
    for child in children:
        if child in visited:
            continue
        logger.debug ("New (unvisited) node: " + child + " must be explored")
        explore(graph, child, visited, myStack)

    return myStack


def explore(graph, node, visited, myStack):
    logger.debug ("Explore: " + node)

    visited.add(node)

    logger.debug ("Explore visited: " + str(visited))
    
    children = graph.get(node)
    if children == None or len(children) == 0:
        return

    logger.debug ("Explore children: " + str(children))
    for child in children:
        if child in visited:
            continue
        logger.debug ("Explore New (unvisited) node: " + child + " must be explored")
        explore(graph, child, visited, myStack)


    myStack.appendleft(node)



def createGraph():
    adjList = {}
    adjList["A"] = set(["C"])
    adjList["B"] = set(["C", "E"])
    adjList["C"] = set(["D"])
    adjList["D"] = set(["F"])
    adjList["E"] = set(["F"])
    adjList["F"] = set(["G"])
    adjList["G"] = None

    return adjList

def main():
    parser = argparse.ArgumentParser(description="Implementing topologicalSort ")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    graph = createGraph()
    topoSorted = runTopologicalSort(graph, "A")
    
    logger.info ("Topologically sorted order: ")
    for e in topoSorted:
        logger.info(e)

if __name__ == '__main__':
    main()
