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



#define INF_COST 0x7fffffff

/** Per-node state. */
class BfsVertexValue {
public:
    int level;
};

static int * level_g;

__device__
BfsVertexValue init_value(BfsVertexValue value) {
    value.level = INF_COST;
    return value; 
}

__device__
BfsVertexValue init_source(BfsVertexValue value) {
    value.level = 0;
    return value;
}

typedef BfsVertexValue (*F)(BfsVertexValue);
__device__ F init_value_p = init_value;
__device__ F init_source_p = init_source;


void aggr_state(VertexId globalIds, BfsVertexValue state) {
    level_g[globalIds] = state.level;
}

int main(int argc, char **argv) {

    if (argc < 2) {
        printf("wrong argument");
        return 1;
    }

    flex::Graph<int, int> graph;
    graph.fromEdgeListFile(argv[1]);
    RandomEdgeCut random;
    auto subgraphs = graph.partitionBy(random, 1);
    Partition<int, int> par;
    par.fromSubgraph(subgraphs[0]);
    auto serial_levels = bfs_serial(par, graph.nodes(), 0);


    Engine<BfsVertexValue, int> engine;
    engine.init(argv[1], 2);
    // The final result, which will be aggregated.
    level_g = (int *) malloc(sizeof(int) * engine.getVertexCount());

    F f_h;
    CUDA_CHECK(cudaMemcpyFromSymbol(&f_h, init_value_p, sizeof(F)));
    engine.vertexMap(f_h);

    CUDA_CHECK(cudaMemcpyFromSymbol(&f_h, init_source_p, sizeof(F)));
    engine.vertexFilter(0, f_h);
    engine.run();
    engine.aggregate(aggr_state);
    auto dist_levels = std::vector<int>(level_g,
                                        level_g + engine.getVertexCount());

    expect_equal(serial_levels, dist_levels);

    printf("Hopefully, nothing goes wrong.");
    return 0;
}
