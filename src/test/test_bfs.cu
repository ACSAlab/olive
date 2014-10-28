/**
 * Test the bfs implementation
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-28
 * Last Modified: 2014-10-28
 */

// totem includes
#include "totem_graph.h"

#define DATA_ROOT(graph_file) "../data/"graph_file

int main(int argc, char** argv) {
    Graph g;
    g.initialize(DATA_ROOT("merrill.csr"));
    g.print();
    g.finialize();
    return 0;
}
