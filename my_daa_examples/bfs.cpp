#include <iostream>
#include <vector>
#include <set>
#include <deque>

void printSet(std::set<int> v)
{
    std::cout << "\nVisited: ";
    for (auto i = v.begin(); i != v.end(); ++i)
        std::cout << *i << " , ";
    std::cout << "\n";
}

void printFrontier(std::deque<int>v)
{
    std::cout << "\nFrontier: ";
    for (auto i = v.begin(); i != v.end(); ++i)
        std::cout << *i << " , ";
    std::cout << "\n";
}

void runBfs(std::vector<std::vector<int>>g)
{
    std::set<int> visited;
    std::deque<int> frontier;
    
    frontier.push_back(0);
    visited.insert(0);

    while(!frontier.empty() && visited.size() != g.size() )
    {
        //get adjacent
        auto current = frontier.front();
        std::cout << "current: " << current << std::endl;
        frontier.pop_front();
        
        visited.insert(current);
        printSet(visited);

        auto adjacent = g.at(current);

        for (auto i = adjacent.begin(); i != adjacent.end(); ++i)
        {
            bool isVisited = visited.find(*i) != visited.end();
            std::cout << *i <<" is " << isVisited << " isVisited?! \n";
            if (!isVisited)
            {
                //std::cout << "Not visited adjacent: " << *i << std::endl;
                frontier.push_back(*i);
            }
        }
        printFrontier(frontier);

    }
}
int main()
{
    std::cout << "BFS\n";
    //create a graph
    std::vector<std::vector<int> > g;
    //0-> 1, 2
    //1-> 2, 3
    //2-> 5
    //3-> 4
    //4-> 1
    std::vector<int>zero;
    zero.push_back(1);
    zero.push_back(2);
    g.push_back(zero);

    std::vector<int>one;
    one.push_back(2);
    one.push_back(3);
    g.push_back(one);

    std::vector<int>two;
    two.push_back(5);
    g.push_back(two);

    std::vector<int>three;
    three.push_back(4);
    g.push_back(three);

    std::vector<int>four;
    four.push_back(1);
    g.push_back(four);

    std::vector<int>five;
    g.push_back(five);

    int counter = 0;
    /*for (auto i = g.begin(); i != g.end(); ++i, ++counter)
    {
        std::cout << "\nNode: " << counter << " : ";
        for (auto j = i->begin(); j != i->end(); ++j)
        {
            std::cout << *j << " \t ";
        }
    }*/

    runBfs(g);
}
