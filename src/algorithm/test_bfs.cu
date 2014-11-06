/**
 * Test the bfs implementation
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-28
 * Last Modified: 2014-11-04
 */

// olive includes
#include "GraphH.h"
#include "GraphD.h"
#include "util.h"

#include <stdio.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("wrong argument");
        return 1;
    }

    printf("device count: %d\n", getGpuNum());
    checkAvailableMemory();

    GraphH graphH;
    graphH.fromFile(argv[1]);
    graphH.print();

    GraphD graphD;
    graphD.fromGraphH(graphH);


    return 0;
}
