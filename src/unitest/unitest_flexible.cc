/**
 * Unit test for the flexible graph representation
 * 
 *
 * Created by: onesuper (onesuperclark@gmail.com)
 * Created on: 2014-11-15
 * Last Modified: 2014-11-15
 */

#include <iostream>
#include <vector>

#include "flexible.h"
#include "unitest_common.h"
#include "partition_strategy.h"


void unitest_sort_and_shuffle(flex::Graph<int, int> &graph) {
    std::cout << "print Scatter after sort\n";
    graph.sortVerticesById();
    graph.printScatter();

    std::cout << "print Scatter after shuffle\n";
    graph.shuffleVertices();
    graph.printScatter();

    std::cout << "print Scatter after shuffleEdges\n";
    graph.shuffleEdges();
    graph.printScatter();

    std::cout << "print Scatter after sortEdges\n";
    graph.sortEdgesById();
    graph.printScatter();

    std::cout << "print Scatter after sort\n";
    graph.sortVerticesById();
    graph.printScatter();
}

void unitest_random_paritition(const flex::Graph<int, int> &graph, int numPart) {
    std::cout << "\n\n\npartition into " << numPart << " parts\n";
    RandomEdgeCut random;
    std::vector<flex::Graph<int, int>> subgraphs = graph.partitionBy(random, numPart);
    for (int i = 0; i < numPart; i++) {
        std::cout << "\n****\n"<< "Partition: " << subgraphs[i].partitionId << std::endl;
        subgraphs[i].printScatter();
        subgraphs[i].printGhostVertices();
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("wrong argument");
        return 1;
    }

    flex::Graph<int, int> graph;
    graph.fromEdgeListFile(argv[1]);

    // Basic Information
    std::cout << "nodes:" << graph.nodes() << std::endl;
    std::cout << "edges:" << graph.edges() << std::endl;
    std::cout << "averageDegree:" << graph.averageDegree() << std::endl;
    std::cout << "printScatter\n";
    graph.printScatter();
    graph.printGhostVertices(); // Supposed to print empty

    // Sort and shuffle vertices and edges
    //unitest_sort_and_shuffle(graph);

    // Tests random parition
    unitest_random_paritition(graph, 4);
    unitest_random_paritition(graph, 3);
    unitest_random_paritition(graph, 2);
    unitest_random_paritition(graph, 1);

    // Shuffle!
    graph.shuffleVertices();
    graph.shuffleEdges();

    // Tests random parition after shuffle
    unitest_random_paritition(graph, 4);

    return 0;
}
