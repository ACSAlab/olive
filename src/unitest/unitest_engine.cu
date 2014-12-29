/**
 * Unit test for the partition
 * 
 *
 * Created by: onesuper (onesuperclark@gmail.com)
 * Created on: 2014-11-23
 * Last Modified: 2014-11-23
 */

#include <iostream>

#include "engine.h"
#include "unitest_common.h"

int main(int argc, char ** argv) {

    if (argc < 3) {
        printf("wrong argument");
        return 1;
    }

    Engine engine;
    engine.init(argv[1], atoi(argv[2]));

    state_g.levels_h = (int *) malloc(sizeof(int) * engine.getVertexCount());

    engine.run();


    for (int i = 0; i < engine.getVertexCount(); i++) {
        printf("%d ", state_g.levels_h[i]);
    }

    return 0;
}
