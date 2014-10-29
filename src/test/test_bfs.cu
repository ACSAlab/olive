/**
 * Test the bfs implementation
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-28
 * Last Modified: 2014-10-28
 */

// olive includes
#include "olive_graph.h"

#define GRAPH_ROOT(graph_file) "../data/"graph_file

int main(int argc, char** argv) {
    Graph g;
    g.initialize(GRAPH_ROOT("merrill.csr"));
    g.print();
    g.finalize();
    return 0;
}
