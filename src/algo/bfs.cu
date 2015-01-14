/**
 * Test the bfs implementation
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-28
 * Last Modified: 2014-12-18
 */

#include "common.h"
#include "bfs_top_down.h"
#include "bfs_serial.h"
#include "bfs_olive.h"

int main(int argc, char **argv) {

    if (argc < 3) {
        printf("wrong argument");
        return 1;
    }

    flex::Graph<int, int> graph;
    graph.fromEdgeListFile(argv[1]);
    RandomEdgeCut random;
    auto subgraphs = graph.partitionBy(random, 1);
    Partition<int, int> single_par;
    single_par.fromSubgraph(subgraphs[0]);
    VertexId src = atoi(argv[2]);

    auto serial_levels = bfs_serial(single_par, graph.nodes(), src);
    auto top_down_levels = bfs_top_down(single_par, graph.nodes(), src);
    auto olive_levels = bfs_olive(argv[1], 2, src);
    
    expect_equal(serial_levels, top_down_levels);
    expect_equal(serial_levels, olive_levels);

    printf("Hopefully, nothing goes wrong.");
    return 0;
}
