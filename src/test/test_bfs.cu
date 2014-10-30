/**
 * Test the bfs implementation
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-28
 * Last Modified: 2014-10-28
 */

// olive includes
#include "olive_graph.h"
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("wrong argument");
        return 1;
    }

    Graph g;
    g.initialize(argv[1]);
    g.print();
    g.finalize();
    return 0;
}
