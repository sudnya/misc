###############################################################################
#
# \file    bfs.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Saturday Aug 27, 2016
# \brief   breadth first search
#
###############################################################################

import argparse
import logging
from collections import deque

logger = logging.getLogger("BFS")

def runBfs(graph, startingNode):
    frontier = deque()
    frontier.append(startingNode)

    visited = set()

    while len(frontier) != 0:
        logger.info("Frontier contains: " + str(frontier))
        logger.info("visited contains: " + str(visited))
        node = frontier.popleft()
        logger.info ("Currently processing: " + str(node))
        
        visited.add(node)
        adjNodes = graph.get(node)
        if adjNodes == None:
            continue

        for entry in adjNodes:
            if entry not in visited:
                logger.info("Adding new (unvisited) connected node: " + entry + " to frontier list")
                frontier.append(entry)
            else:
                logger.info("Already saw: " + entry + " do nothing")


def createGraph():
    adjList = {}
    adjList["A"] = set(["B", "C"])
    adjList["B"] = set("C")
    adjList["C"] = set("D")
    adjList["D"] = set("E")
    adjList["E"] = set(["B", "F"])
    adjList["F"] = None

    return adjList

def main():
    parser = argparse.ArgumentParser(description="Implementing BFS ")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    graph = createGraph()
    runBfs(graph, "F")

if __name__ == '__main__':
    main()
