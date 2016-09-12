###############################################################################
#
# \file    dijkstra.py
# \author  Sudnya Padalikar <mailsudnya@gmail.com>
# \date    Sunday Sept 11, 2016
# \brief   Dijkstra's single source shortest path algorithm
#
###############################################################################

import argparse
import logging
import sys

import Queue as Q

logger = logging.getLogger(" Dijsktra ")

def runDijkstra(graph, startingNode):
    frontier = Q.PriorityQueue()
    frontier.put(startingNode, 0)

    parent = {}
    cost = {}

    initializeCost(cost, startingNode, graph.keys())

    while not frontier.empty():
        node = frontier.get()

        logger.debug ("Considering node: " + node)
        if node == None or len(node) == 0:
            break

        neighbors = graph.get(node)
        if neighbors == None or len(neighbors) == 0:
            continue

        for neighbor in neighbors:
            newCost = cost.get(node) + evalCost(node, neighbor)
            if neighbor not in cost or cost.get(neighbor) > newCost:
                logger.debug("Neighbor " + neighbor + " has better cost: " + str(newCost))
                cost[neighbor] = newCost
                parent[neighbor] = node
                frontier.put(neighbor, newCost)
                logger.debug("Parent list: " + str(parent))
                logger.debug("Cost list: " + str(cost))

def initializeCost(costMap, startNode, nodes):
    for node in nodes:
        if node == startNode:
            costMap[node] = 0
        else:
            costMap[node] = sys.maxint

def evalCost(a, b):
    costMap = {}

    costMap[("A", "B")] = 5
    costMap[("B", "A")] = 5
    
    costMap[("B", "C")] = 2
    costMap[("C", "B")] = 2

    costMap[("C", "D")] = 3
    costMap[("D", "C")] = 3

    costMap[("D", "F")] = 2
    costMap[("F", "D")] = 2

    costMap[("D", "A")] = 9
    costMap[("A", "D")] = 9

    costMap[("F", "E")] = 3
    costMap[("E", "F")] = 3

    costMap[("E", "A")] = 2
    costMap[("A", "E")] = 2

    return costMap[(a,b)]

def createGraph():
    adjList = {}
    adjList["A"] = set(["B", "E"])
    adjList["B"] = set(["A", "C"])
    adjList["C"] = set(["B", "D"])
    adjList["D"] = set(["C", "F"])
    adjList["E"] = set(["A", "F"])
    adjList["F"] = set(["D", "E"])

    return adjList

def main():
    parser = argparse.ArgumentParser(description="Implementing Dijkstra ")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true")

    parsedArguments = parser.parse_args()
    arguments = vars(parsedArguments)

    isVerbose   = arguments['verbose']

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    graph = createGraph()
    runDijkstra(graph, "A")

if __name__ == '__main__':
    main()
