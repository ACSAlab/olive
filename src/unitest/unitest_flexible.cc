/**
 * Unit test for the flexible graph representation
 * 
 *
 * Created by: onesuper (onesuperclark@gmail.com)
 * Created on: 2014-11-15
 * Last Modified: 2014-11-15
 */

#include <iostream>
#include "flexible.h"
#include "unitest_common.h"



int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("wrong argument");
        return 1;
    }

    flex::Graph<int, int> graph;

    graph.fromEdgeListFile(argv[1]);

    std::cout << "nodes:" << graph.nodes() << std::endl;

    std::cout << "edges:" << graph.edges() << std::endl;

    std::cout << "averageDegree:" << graph.averageDegree() << std::endl;


    std::cout << "printScatter\n";
    graph.printScatter();

    std::cout << "printGather\n";
    graph.printGather();

    std::cout << "printScatter withAttr\n";
    graph.printScatter(true);

    std::cout << "printGather withAttr\n";
    graph.printGather(true);

    std::cout << "print Scatter after sort\n";
    graph.sort();
    graph.printScatter();

    std::cout << "print Scatter after shuffile\n";
    graph.shuffle();
    graph.printScatter();

    std::cout << "print Scatter after shuffleEdges\n";
    graph.shuffleEdges();
    graph.printScatter();

    std::cout << "print Scatter after SortEdges\n";
    graph.sortEdges();
    graph.printScatter();

    std::cout << "print Scatter after sort\n";
    graph.sort();
    graph.printScatter();

    graph.printDegreeDist();



    return 0;
}
