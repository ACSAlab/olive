/**
 * Test the bfs implementation
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-28
 * Last Modified: 2014-12-18
 */

#include "partition.h"
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

    Partition par;
    par.fromSubgraph(subgraphs[0]);

    auto top_down_levels = bfs_top_down(par, graph.nodes(), 0);
    auto serial_levels = bfs_top_down(par, graph.nodes(), 0);

    expect_equal(top_down_levels, serial_levels);

    printf("Hopefully, nothing goes wrong.");
    return 0;
}
