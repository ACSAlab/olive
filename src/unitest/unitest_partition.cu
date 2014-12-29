/**
 * Unit test for the partition
 * 
 *
 * Created by: onesuper (onesuperclark@gmail.com)
 * Created on: 2014-11-23
 * Last Modified: 2014-11-23
 */

#include <iostream>
#include <vector>

#include "partition.h"
#include "unitest_common.h"

void print_paritition(const Partition& partition) {

    std::cout << "\n****\n"<< "Partition: " << partition.partitionId
        << "/" << partition.numParts << std::endl;

    printf("\nvertices: ");
    for (size_t i = 0 ; i < partition.vertices.size(); i++) {
        std::cout << partition.vertices[i] << " "; 
    }
    printf("\nedges: ");
    for (size_t i = 0 ; i < partition.edges.size(); i++) {
        std::cout  << i << ":" << partition.edges[i].partitionId << "-" 
                  << partition.edges[i].localId << " " ; 
    }
    printf("\nglobalIds: ");
    for (size_t i = 0 ; i < partition.globalIds.size(); i++) {
        std::cout << partition.globalIds[i] << " "; 
    }
    printf("\n");
}

int main(int argc, char ** argv) {

    if (argc < 3) {
        printf("wrong argument");
        return 1;
    }

    flex::Graph<int, int> graph;
    PartitionId numParts = atoi(argv[2]);
    graph.fromEdgeListFile(argv[1]);
    // graph.printOutEdges();
    // graph.printGhostVertices();

    RandomEdgeCut random;
    auto subgraphs = graph.partitionBy(random, numParts);
    for (int i = 0; i < numParts; i++) {
        std::cout << "\n****\n"<< "Subgraph: " << subgraphs[i].partitionId
            << "/" << subgraphs[i].numParts << std::endl;
        subgraphs[i].printOutEdges();
        subgraphs[i].printGhostVertices();
    }

    std::vector<Partition> pars;
    pars.resize(numParts);
    for (int i = 0; i < numParts; i++) {
        pars[i].fromSubgraph(subgraphs[i]);
        print_paritition(pars[i]);
    }

    return 0;
}
