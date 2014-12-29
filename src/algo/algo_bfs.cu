/**
 * Test the bfs implementation
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-28
 * Last Modified: 2014-12-18
 */

#include "engine.h"
#include "algo_common.h"
#include "bfs_top_down.h"
#include "bfs_serial.h"

int main(int argc, char **argv) {

    if (argc < 2) {
        printf("wrong argument");
        return 1;
    }

    flex::Graph<int, int> graph;
    graph.fromEdgeListFile(argv[1]);
    // Here we just utilize one partition to test whether the GRD representation
    // works.
    RandomEdgeCut random;
    auto subgraphs = graph.partitionBy(random, 1);

    Partition<int> par;
    par.fromSubgraph(subgraphs[0]);
    auto serial_levels = bfs_top_down(par, graph.nodes(), 0);


    Engine<int> engine;
    engine.init(argv[1], 2);
    state_g.levels_h = (int *) malloc(sizeof(int) * engine.getVertexCount());
    engine.run();

    auto dist_levels = std::vector<int>(state_g.levels_h,
                                        state_g.levels_h + engine.getVertexCount());


    expect_equal(serial_levels, dist_levels);

    printf("Hopefully, nothing goes wrong.");
    return 0;
}
