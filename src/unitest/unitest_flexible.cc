/**
 * Unit test for the flexible graph representation
 * 
 *
 * Created by: onesuper (onesuperclark@gmail.com)
 * Created on: 2014-11-15
 * Last Modified: 2014-11-15
 */

#include "flexible.h"

#include "unitest_common.h"



int main(int argc, char ** arg) {
    if (argc < 2) {
        printf("wrong argument");
        return 1;
    }

    flex::Graph<int, int> graph;

    graph.fromEdgeListFile(argv[1]);


    return 0;
}
