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

#include "csrGraph.h"
#include "commandLine.h"


int main(int argc, char **argv) {

    CommandLine cl(argc, argv, "<inFile> [-part 2] [-verbose]");
    char * inFile = cl.getArgument(0);
    int numParts = cl.getOptionIntValue("-part", 1);
    bool quiet = cl.getOption("-verbose");

    CsrGraph<int, int> graph;
    graph.fromEdgeListFile(inFile);

    // Basic Information
    std::cout << "Nodes:" << graph.vertexCount;
    std::cout << " Edges:" << graph.edgeCount << std::endl;

    if (!verbose) {
        return 0;
    }

    graph.printOutEdges();    
    graph.printInEdges();
    graph.printDegreeHistogram();
    graph.printDegreeHistogram(false);

    return 0;
}
