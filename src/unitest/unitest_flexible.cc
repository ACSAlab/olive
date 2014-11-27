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
    graph.print();

    std::cout << "print Scatter after shuffle\n";
    graph.shuffleVertices();
    graph.print();

    std::cout << "print Scatter after shuffleEdges\n";
    graph.shuffleEdges();
    graph.print();

    std::cout << "print Scatter after sortEdges\n";
    graph.sortEdgesById();
    graph.print();

    std::cout << "print Scatter after sort\n";
    graph.sortVerticesById();
    graph.print();
}

void unitest_random_paritition(const flex::Graph<int, int> &graph, int numPart) {
    std::cout << "\nBefore partitioning " << numPart << " parts\n";
    graph.print();
    graph.printGhostVertices();
    std::cout << "\npartition into " << numPart << " parts\n";
    RandomEdgeCut random;
    auto subgraphs = graph.partitionBy(random, numPart);
    for (int i = 0; i < numPart; i++) {
        std::cout << "\n****\n"<< "Partition: " << subgraphs[i].partitionId << std::endl;
        subgraphs[i].print();
        subgraphs[i].printGhostVertices();
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("wrong argument");
        return 1;
    }

    flex::Graph<int, int> graph;
    graph.fromEdgeListFile(argv[1]);
    PartitionId numParts = atoi(argv[2]);
    // Basic Information
    std::cout << "nodes:" << graph.nodes() << std::endl;
    std::cout << "edges:" << graph.edges() << std::endl;
    std::cout << "averageDegree:" << graph.averageDegree() << std::endl;
    std::cout << "print\n";
    graph.print();
    graph.printGhostVertices(); // Supposed to print empty

    // Sort and shuffle vertices and edges
    //unitest_sort_and_shuffle(graph);


    // Shuffle!
    graph.shuffleVertices();
    graph.shuffleEdges();

    // Tests random parition after shuffle
    unitest_random_paritition(graph, numParts);

    return 0;
}
