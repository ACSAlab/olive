/**
 * Unit test for the partition
 * 
 *
 * Created by: onesuper (onesuperclark@gmail.com)
 * Created on: 2014-11-23
 * Last Modified: 2014-11-23
 */

#include "unitest_common.h"
#include "partition.h"

#include <iostream>

void print_paritition(const Partition& partition) {
    printf("\nvertices: ");
    for (size_t i = 0 ; i < partition.vertices.size(); i++) {
        std::cout << partition.vertices[i] << " "; 
    }
    printf("\nedges: ");
    for (size_t i = 0 ; i < partition.edges.size(); i++) {
        std::cout  << i << ":" << partition.edges[i].partitionId << "-" 
                  << partition.edges[i].id << " " ; 
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
    graph.print();
    graph.printGhostVertices();


    RandomEdgeCut random;
    auto subgraphs = graph.partitionBy(random, numParts);
    for (int i = 0; i < numParts; i++) {
        std::cout << "\n****\n"<< "Partition: " << subgraphs[i].partitionId << std::endl;
        subgraphs[i].print();
        subgraphs[i].printGhostVertices();
    }

    Partition par[4];
    for (int i = 0; i < numParts; i++) {
        std::cout << "\n****\n"<< "Partition: " << subgraphs[i].partitionId << std::endl;
        par[i].fromSubgraph(subgraphs[i]);
        print_paritition(par[i]);
    }


    return 0;
}
