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

#include "olive.h"

void unitest_random_paritition(flex::Graph<int, int> &graph, int numPart) {
    std::cout << "\nBefore partitioning " << numPart << " parts\n";
    graph.printOutEdges();
    graph.printGhostVertices();
    graph.printInEdges();

    std::cout << "\npartition into " << numPart << " parts\n";
    RandomEdgeCut random;
    auto subgraphs = graph.partitionBy(random, numPart);
    for (int i = 0; i < numPart; i++) {
        std::cout << "\n****\n" << "Partition: " << subgraphs[i].partitionId << std::endl;
        subgraphs[i].printOutEdges();
        subgraphs[i].printGhostVertices();
        subgraphs[i].printInEdges();
    }
}

int main(int argc, char **argv) {

    CommandLine cl(argc, argv, "<inFile> [-part 2] [-shuffle] [-quiet]");
    char * inFile = cl.getArgument(0);
    int numParts = cl.getOptionIntValue("-part", 1);
    bool shuffle = cl.getOption("-shuffle");
    bool quiet = cl.getOption("-quiet");

    flex::Graph<int, int> graph;
    graph.fromEdgeListFile(inFile);

    // Basic Information
    std::cout << "nodes:" << graph.nodes() << std::endl;
    std::cout << "edges:" << graph.edges() << std::endl;
    std::cout << "averageDegree:" << graph.averageDegree() << std::endl;

    if (quiet) {
        return 0;
    }


    // Shuffle!
    if (shuffle) {
        graph.shuffleVertices();
        graph.shuffleEdges();
    }

    // Tests random parition after shuffle
    unitest_random_paritition(graph, numParts);

    return 0;
}
