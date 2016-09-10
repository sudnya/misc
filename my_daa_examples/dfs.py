###############################################################################
#
# \file    dfs.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Saturday Aug 27, 2016
# \brief   depth first search
#
###############################################################################

import argparse
import logging
from collections import deque

logger = logging.getLogger("BFS")

def runDfs(graph):
    frontier = []
    visited = set()

    for k, v in graph.iteritems():
        frontier.append(k)
        
        while len(visited) < len(graph.keys()):
            if len(frontier) == 0:
                break

            j = frontier.pop()
            logger.info("Popped " + j + " for processing")

            visited.add(j)
            logger.info("Visited contains: " + str(visited))

            connected = graph.get(j)
            
            if connected == None:
                continue

            for entry in connected:
                if entry not in visited:
                    logger.info("Added " + entry + " to frontier")
                    frontier.append(entry)
                    logger.info("Frontier contains: " + str(frontier))


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
    runDfs(graph)

if __name__ == '__main__':
    main()
