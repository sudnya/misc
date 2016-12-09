/*
 *
 * \file    bfs.cpp
 * \author  Sudnya Diamos <mailsudnya@gmail.com>
 * \date    Friday Dec 9, 2016
 * \brief   breadth first search
 *
**/

#include <iostream>
#include <vector>
#include <set>
#include <deque>

template <typename T>
void printSet(const T v)
{
    for (auto& i : v)
        std::cout << i << " , ";
    std::cout << "\n";
}

template <typename T>
void runBfs(T g) 
{
    std::set<int> visited;
    std::deque<int> frontier;
    
    frontier.push_back(0);
    visited.insert(0);

    while(!frontier.empty() && visited.size() != g.size() )
    {
        //get adjacent
        auto current = frontier.front();
        std::cout << "visiting current node: " << current << std::endl;
        frontier.pop_front();
        
        visited.insert(current);

        auto adjacent = g.at(current);

        for (auto i = adjacent.begin(); i != adjacent.end(); ++i)
        {
            bool isVisited = visited.find(*i) != visited.end();
            if (!isVisited)
            {
                std::cout << "adding " << *i << " to frontier\n";
                frontier.push_back(*i);
            }
        }
        std::cout << "frontier: ";
        printSet(frontier);
        std::cout << "visited: ";
        printSet(visited);
    }
}

template <typename T>
void printInput(T g)
{
    int counter = 0;
    for (auto i = g.begin(); i != g.end(); ++i, ++counter)
    {
        std::cout << "\nNode: " << counter << " : ";
        for (auto j = i->begin(); j != i->end(); ++j)
        {
            std::cout << *j << " \t ";
        }
    }

}
/*
 *      0 -> 1 ->   3
 *     \|  .  /.\  \|/
 * 5 <- 2 \./     .4
 *
 */
int main()
{
    std::cout << "BFS\n";
    //create a graph
    //0-> 1, 2
    //1-> 2, 3
    //2-> 5
    //3-> 4
    //4-> 1
    std::vector<int>zero  = {1,2};
    std::vector<int>one   = {2,3};
    std::vector<int>two   = {5};
    std::vector<int>three = {4};
    std::vector<int>four  = {1};
    std::vector<int>five  = {};

    std::vector<std::vector<int>> g = {zero, one, two, three, four, five};

    //printInput(g);

    runBfs(g);
}
