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

    CommandLine cl(argc, argv, "<inFile> [-dimacs] [-verbose]");
    char * inFile = cl.getArgument(0);
    bool dimacs = cl.getOption("-dimacs"); 
    bool verbose = cl.getOption("-verbose");

    CsrGraph<int, int> graph;
    if (dimacs) {
        graph.fromDimacsFile(inFile);
    } else {
        graph.fromEdgeListFile(inFile);
    }

    // Basic Information
    std::cout << "Nodes:" << graph.vertexCount;
    std::cout << " Edges:" << graph.edgeCount << std::endl;

    if (verbose) graph.print(false);    
    return 0;
}
